------------------------------------------------------------------------
--[[ SequencerCriterion ]]--
-- Applies a criterion to each of the inputs and targets in the 
-- corresponding input and target Tables.
-- Useful for nn.Repeater and nn.Sequencer.
-- WARNING : assumes that the decorated criterion is stateless, i.e. 
-- the backward doesn't need to be preceded by a commensurate forward.
------------------------------------------------------------------------
local SequencerCriterion, parent = torch.class('nn.SequencerCriterion', 'nn.Criterion')

function SequencerCriterion:__init(criterion)
   parent.__init(self)
   self.criterion = criterion
   if torch.isTypeOf(criterion, 'nn.ModuleCriterion') then
      error("SequencerCriterion shouldn't decorate a ModuleCriterion. "..
         "Instead, try the other way around : "..
         "ModuleCriterion decorates a SequencerCriterion. "..
         "Its modules can also be similarly decorated with a Sequencer.")
   end
   self.gradInput = {}
   self._gradInput = {}
end

function SequencerCriterion:updateOutput(inputTable, targetTable)
   self.output = 0
   for i,input in ipairs(inputTable) do
      self.output = self.output + self.criterion:forward(input, targetTable[i])
   end
   return self.output
end

function SequencerCriterion:updateGradInput(inputTable, targetTable)
   self.gradInput = {}
   for i,input in ipairs(inputTable) do
     self.gradInput[i] = self.criterion:backward(input, targetTable[i])
     for j = 1, targetTable[i]:size()[1] do -- set the gradients of '#' to zero
      if targetTable[i][j] == ds.vocab_size then
	    self.gradInput[i][j][ds.vocab_size] = 0 -- set to zero
      end
     end
   end
   return self.gradInput
end
