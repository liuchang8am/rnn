local RecurrentCaption, parent = torch.class("nn.RecurrentCaption", "nn.AbstractSequencer")

function RecurrentCaption:__init(rnn, action, nStep, hiddenSize, cuda)
    parent.__init(self)
    assert(torch.isTypeOf(action, 'nn.Module'))
    assert(torch.type(nStep) == 'number')
    assert(torch.type(hiddenSize) == 'table')
    assert(torch.type(hiddenSize[1]) == 'number', "Does not support table hidden layers")

    self.rnn = rnn
    -- we can decorate the module with a Recursor to make it AbstractRecurrent
    self.rnn = (not torch.isTypeOf(rnn, 'nn.AbstractRecurrent')) and nn.Recursor(rnn) or rnn
    -- backprop through time (BPTT) will be done online (in reverse order of forward)
    self.rnn:backwardOnline()
    for i, modula in ipairs(self.rnn:listModules()) do
        if torch.isTypeOf(modula, "nn.AbstractRecurrent") then
            modula.copyInputs = false
            modula.copyGradOutputs = false
        end
    end

    -- samples an x,y actions for each example
    self.action = (not torch.isTypeOf(action, 'nn.AbstractRecurrent')) and nn.Recursor(action) or action
    self.action:backwardOnline()
    self.hiddenSize = hiddenSize
    self.nStep = nStep

    self.modules = { self.rnn, self.action }

    self.output = {} -- rnn output
    self.actions = {} -- action output

    self.forwardActions = false

    self.gradHidden = {}
    self._cuda = cuda
end

function RecurrentCaption:updateOutput(input)
    self.rnn:forget()
    self.action:forget()
    local nDim = input:dim()
    self.output = {}
    for step = 1, self.nStep do
        if step == 1 then
            -- sample an initial starting actions by forwarding zeros through the action
            self._initInput = self._initInput or input.new()
            self._initInput:resize(input:size(1), table.unpack(self.hiddenSize)):zero()
            self.actions[1] = self.action:updateOutput(self._initInput)
        else
            -- sample actions from previous hidden activation (rnn output)
            self.actions[step] = self.action:updateOutput(self.output[step - 1])
        end

        -- rnn handles the recurrence internally
        local output = self.rnn:updateOutput { input, self.actions[step] }
        self.output[step] = self.forwardActions and { output, self.actions[step] } or output
    end
    
    -- convert the table output format to doubletensor output format
    if (self._cuda) then
      temp_output = torch.Tensor(self.nStep, self.output[1][1]:size()[1]):cuda()
      for k,v in pairs(self.output) do
        temp_output[k] = v[1]:cuda()
      end
    else
      temp_output = torch.Tensor(self.nStep, self.output[1][1]:size()[1]) -- rho x hiddenSize doubletensor
      for k, v in pairs(self.output) do
        temp_output[k] = v[1]
      end
    end
    
    self.output = temp_output
    return self.output
end

function RecurrentCaption:updateGradInput(input, gradOutput)
    assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
    --assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
    --assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    assert(gradOutput:size()[1] == self.nStep, "gradOutput should have nStep elements")

    -- back-propagate through time (BPTT)
    for step = self.nStep, 1, -1 do
        -- 1. backward through the action layer
        local temp_gradOutput
        if self._cuda then
          temp_gradOutput = torch.Tensor(1,gradOutput:size()[2]):cuda()
          temp_gradOutput[1] = gradOutput[step]:cuda()
        else
          temp_gradOutput = torch.Tensor(1,gradOutput:size()[2])
          temp_gradOutput[1] = gradOutput[step]
        end
        --local gradOutput_, gradAction_ = gradOutput[step]
        local gradOutput_, gradAction_ = temp_gradOutput
        if self.forwardActions then
            gradOutput_, gradAction_ = unpack(gradOutput[step])
        else
            -- Note : gradOutput is ignored by REINFORCE modules so we give a zero Tensor instead
            self._gradAction = self._gradAction or self.action.output.new()
            if not self._gradAction:isSameSizeAs(self.action.output) then
                self._gradAction:resizeAs(self.action.output):zero()
            end
            gradAction_ = self._gradAction
        end

        if step == self.nStep then
            self.gradHidden[step] = nn.rnn.recursiveCopy(self.gradHidden[step], gradOutput_)
        else
            -- gradHidden = gradOutput + gradAction
            nn.rnn.recursiveAdd(self.gradHidden[step], gradOutput_)
        end

        if step == 1 then
            -- backward through initial starting actions
            self.action:updateGradInput(self._initInput, gradAction_)
        else
            local gradAction = self.action:updateGradInput(self.output[step - 1], gradAction_)
            self.gradHidden[step - 1] = nn.rnn.recursiveCopy(self.gradHidden[step - 1], gradAction)
        end

        -- 2. backward through the rnn layer
        local gradInput = self.rnn:updateGradInput(input, self.gradHidden[step])[1]
        if step == self.nStep then
            self.gradInput:resizeAs(gradInput):copy(gradInput)
        else
            self.gradInput:add(gradInput)
        end
    end

    return self.gradInput
end

function RecurrentCaption:accGradParameters(input, gradOutput, scale)
    assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
    --assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
    --assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    assert(gradOutput:size()[1] == self.nStep, "gradOutput should have nStep elements")

    -- back-propagate through time (BPTT)
    for step = self.nStep, 1, -1 do
        -- 1. backward through the action layer
        local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction

        if step == 1 then
            -- backward through initial starting actions
            self.action:accGradParameters(self._initInput, gradAction_, scale)
        else
            self.action:accGradParameters(self.output[step - 1], gradAction_, scale)
        end

        -- 2. backward through the rnn layer
        self.rnn:accGradParameters(input, self.gradHidden[step], scale)
    end
end

function RecurrentCaption:accUpdateGradParameters(input, gradOutput, lr)
    assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
    --assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
    --assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    assert(gradOutput:size()[1] == self.nStep, "gradOutput should have nStep elements")

    -- backward through the action layers
    for step = self.nStep, 1, -1 do
        -- 1. backward through the action layer
        local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction

        if step == 1 then
            -- backward through initial starting actions
            self.action:accUpdateGradParameters(self._initInput, gradAction_, lr)
        else
            -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
            self.action:accUpdateGradParameters(self.output[step - 1], gradAction_, lr)
        end

        -- 2. backward through the rnn layer
        self.rnn:accUpdateGradParameters(input, self.gradHidden[step], lr)
    end
end

function RecurrentCaption:type(type)
    self._input = nil
    self._actions = nil
    self._crop = nil
    self._pad = nil
    self._byte = nil
    return parent.type(self, type)
end

function RecurrentCaption:__tostring__()
    local tab = '  '
    local line = '\n'
    local ext = '  |    '
    local extlast = '       '
    local last = '   ... -> '
    local str = torch.type(self)
    str = str .. ' {'
    str = str .. line .. tab .. 'action : ' .. tostring(self.action):gsub(line, line .. tab .. ext)
    str = str .. line .. tab .. 'rnn     : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
    str = str .. line .. '}'
    return str
end
