classdef AdaBoost < handle & Classifier
%WARNING: need further testing!!

	properties
		func; %function handle of the constructer of weak classifier
		par; %parameters for the function handle
		classifiers; %cell array of weak classifiers
		classifier_weights; %weight for weak classifiers (sample weights are inherit from Classifier)		
		numclass;
		max_iter = 5;

		%for debug
		error_bound;
	end
	
	methods
		function self = AdaBoost(func, par)
			self.func = func;			
			self.par = par;
		end
		
		function [] = train(self, X, Y) % Y must be conti. positive integer
			%init weights
			[numdim numsamples] = size(X);
			self.numclass = length(unique(Y));
			self.sample_penalty = ones(numsamples,1)/numsamples;
			self.classifier_weights = zeros(self.max_iter,1);
			
			self.error_bound = zeros(self.max_iter,1);
			for t = 1 : self.max_iter
				switch length(self.par) %should not be too much
				case 0
					self.classifiers{t} = self.func();								
				case 2
					self.classifiers{t} = self.func(self.par{1},self.par{2});								
					% self.classifiers{t}.setClassPenalty(self.class_penalty); 				
				otherwise 
					error('implement the number!');
				end
				self.classifiers{t}.setSamplePenalty(length(Y)*self.sample_penalty);%to avoid numerical problem
				self.classifiers{t}.train(X,Y);
				[pred] = self.classifiers{t}.classify(X);
				
				w = self.sample_penalty;
				if ~isempty(self.class_penalty)
					for i = 1 : length(self.class_penalty)
						w(Y==i) = w(Y==i)*self.class_penalty(i);
					end
				end
				
				self.classifier_weights(t) = log( sum(w(Y==pred)) / sum(w(Y~=pred)) ) + log(self.numclass-1);
				
				%update sample weights				
				self.sample_penalty = w.* exp(-self.classifier_weights(t)*(pred==Y));
				self.error_bound(t) = sum(self.sample_penalty);
				self.sample_penalty = self.sample_penalty / sum(self.sample_penalty);
				
				% [pred] = self.classify(X,Y);
				% fprintf('error rate = %f\n',sum(w(Y~=pred)));
			end
						
		end
		
		function [pred accu] = classify(self, X, Y)
			H = zeros(size(X,2), self.numclass);
			for i = 1 : length(self.classifiers)
				pred_weak = self.classifiers{i}.classify(X);				
				H = H + Utils.num2bin(pred_weak,self.numclass)*self.classifier_weights(i);
			end			
			[~, pred] = max(H,[],2);
			
			accu = [];
			pred = pred(:);
			if exist('Y','var')
				accu = mean(pred == Y);                                                            
			end
		end
	end

end
