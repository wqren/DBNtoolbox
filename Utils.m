classdef Utils
	%general util functions
	methods(Static)
	    function Yout = toConsecutive(y)
	        ylist = unique(y);
	        Yout = zeros(size(y));
	        for i = 1 : length(ylist)
	            Yout(y==ylist(i)) = i;
	        end
	    end
	    
		function Yout = num2bin(y,K)			
		    y = y(:);
			Yout = bsxfun(@(y,ypos) (y==ypos), y, 1:K)';
		end
		
		function Yout = bin2num(y_multi, dim)
		    if dim == 2
		        y_multi = y_multi';
		    end
		    y_multi = bsxfun(@times, y_multi, [1:size(y_multi,1)]');
		    Yout = sum(y_multi,1)';
		end
		
		function X = sigmoid(X)
			X = 1./(1+exp(-X));
		end
        
        function X = vec(X)
            X = X(:);
        end
        
        function [coeff] = pbcorr(X,Y)
            %point-biserial correlation coefficient, X is continuous (numvariables*numsamples), Y is binary
            coeff = sqrt(nnz(Y==1)*nnz(Y==0)/(length(Y)^2)) * (mean(X(:,Y==1),2) - mean(X(:,Y==0),2)) ./ std(X,[],2)
        end
        
        function [d dy dh] = checkgrad2_nodisplay(f, X, e, P1, P2, P3, P4, P5);
        % Carl Edward Rasmussen, 2001-08-01.
        
            [y dy] = feval(f, X);                         % get the partial derivatives dy

            dh = zeros(numel(X),1) ;
            for j = 1:numel(X)
              dx = zeros(numel(X),1);
              dx(j) = dx(j) + e;                               % perturb a single dimension
              y2 = feval(f, X+reshape(dx, size(X)));
              dx = -dx ;
              y1 = feval(f, X+reshape(dx, size(X)));
              dh(j) = (y2 - y1)/(2*e);
            end

            dy = dy(:);
        %     disp([dy dh])                                          % print the two vectors
            d = norm(dh-dy)/norm(dh+dy);       % return norm of diff divided by norm of sum, y : computed, h : numerical
        end
        
        %TODO: move to somewhere else
        function display_network(A, learner_set, numcols, numchannels, figstart)            
            %A = numunits * numsamples
            %learner_set : from high to low
            for i = length(learner_set) : -1 : 1
               A =  learner_set{i}.weights * A;
            end
            if ~exist('numcols', 'var') 
                numcols = [];
            end
            if ~exist('numchannels', 'var') 
                numchannels = [];
            end
            if ~exist('figstart', 'var') 
                 figstart = [];
            end
            DeepBeliefNetwork.display_network_l1(A, numcols, numchannels, figstart);
        end
        
       
        
        %--------function for convolution in CNN-----------------------
        function [C] = convInference(A,B)
            %A = M*N*c*d
            %B = m*n*c*u
            %C = (M-m+1)*(N-n+1)*u*d
            [M N numchannels numdata] = size(A);
            [m n numchannels, numunits] = size(B);
            C = Utils.zeros([M-m+1,N-n+1,numunits,numdata]);
            
            %optimize this further with different loops? 
            for u = 1 : numunits
                for d = 1 : numdata
                    C(:,:,u,d) = convn(A(:,:,:,d),squeeze(B(end:-1:1,end:-1:1,end:-1:1,u)),'valid');
                end
            end

%             for c = 1 : numchannels
%                 for u = 1 : numunits
%                     for d = 1 : numdata
%                         C(:,:,u,d) = C(:,:,u,d) + filter2(B(:,:,c,u), A(:,:,c,d),'valid') ;                                  
%                     end
%                 end
%             end
        end
        
        function [C] = convReconstruct(A,B)
            [M N numunits numdata] = size(A);
            [m n numchannels, numunits] = size(B);
            C = Utils.zeros([M+m-1,N+n-1,numchannels,numdata]);            

            %convn doesn't support full convolution well, how to make this
            %faster?
            for c = 1 : numchannels
                for u = 1 : numunits
                    for d = 1 : numdata                    
                        C(:,:,c,d) = C(:,:,c,d) + conv2(B(:,:,c,u),A(:,:,u,d),'full'); 
                    end
                end
            end    
        end
        
        function [C] = convGradient(A,B)
            [M N numchannels numdata] = size(A);
            [m n numunits, numdata] = size(B);
            C = Utils.zeros([M-m+1,N-n+1,numchannels,numunits]);            
            
%             A = permute(A,[1 2 4 3]);
%             B = permute(B,[1 2 4 3]);
%             for c = 1 : numchannels
%                 for u = 1 : numunits
%                     C(:,:,c,u) = convn(A(:,:,:,c),B(end:-1:1,end:-1:1,end:-1:1,u),'valid');
%                 end
%             end
            
            for c = 1 : numchannels
                for u = 1 : numunits
                    for d = 1 : numdata
                        C(:,:,c,u) = C(:,:,c,u) + filter2(B(:,:,u,d), A(:,:,c,d),'valid');
                    end
                end
            end
        end
                
        %----------------for GPU------------------
        function [X] = zeros(dims)
            if Config.gpu 
                X = gpuArray(zeros(dims));
            else
                X = zeros(dims);
            end
        end
        
        function [X] = ones(dims)
            if Config.gpu 
                X = gpuArray(ones(dims));
            else
                X = ones(dims);
            end
        end
        
        function [X] = rand(dims)
            if Config.gpu 
                X = gpuArray(rand(dims));
            else
                X = rand(dims);
            end
        end
        
        function [X] = randn(dims)
            if Config.gpu 
                X = gpuArray(randn(dims));
            else
                X = randn(dims);
            end
        end
        
        end
    end
