classdef Classifier < handle
	properties		       
        class_penalty; %mis-classification penalty for each class, numclass*1
		sample_penalty; %mis-classification penalty for each sample, numsample*1                
	end
	
	methods
		function self = setClassPenalty(self, class_penalty)
			self.class_penalty = class_penalty(:);		
		end
		
		function self = setSamplePenalty(self, sample_penalty)
			self.sample_penalty = sample_penalty(:);		
		end
		
		function [] = classPenaltyOnSamples(self,y)
		    numsamples = length(y);
            penalty = zeros(numsamples,1);
            numclass = length(self.class_penalty);
            for c = 1 : numclass
                penalty(y==c) = self.class_penalty(c);
            end
            if isempty(self.sample_penalty)
                self.sample_penalty = penalty;
            else
                self.sample_penalty = self.sample_penalty.*penalty;
            end
		end
		
		function [] = train(self,X,Y)
			error('need implement this!')
		end
		
		function [pred] = classify(self,X,Y)
			error('need implement this!')
		end
	end
end