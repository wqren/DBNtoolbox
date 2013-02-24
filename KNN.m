classdef KNN < handle
	%k nearest neighbor
	properties
		K;
		dist;
		
		X; %will matlab really copy that or just keep the pointer?
		Y;
	end
	
	methods
		function self = KNN(K, dist)
			self.K = K;
			
			%not useful now
			if ~exist('dist','var')
				self.dist = @(X1,X2)(sum((X1-X2).^2)); %use euclidean distance by default
			else
				self.dist = dist;
			end
		end
		
		function [] = train(self, X,Y)
			self.X = X;
			self.Y = Y;
		end
		
		function [pred accu] = classify(self, X, Y)
			pred = zeros(size(X,2),1);			
			for i = 1 : size(X,2)
				% if (mod(i,100) == 0)
					% fprintf('[%d/%d]',i,size(X,2));
				% end
				%too slow.. give up this custom distance function idea
				% dist = zeros(size(self.X,2),1);
				% for j = 1 : size(self.X,2)
					% dist(j) = self.dist(self.X(:,j),X(:,i));
				% end		
				dist = sum(bsxfun(@minus,self.X,X(:,i)).^2,1);
				
				[~, idx] = sort(dist, 'ascend');
				pred(i) = mode(self.Y(idx(1:self.K)));
			end
			
			accu = [];		
			if exist('Y','var')
				accu = mean(pred == Y);                                                            
			end
		end
		
	end
end