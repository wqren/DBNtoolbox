classdef FeatureSelector < handle
	%wrapper for multiple feature selection algorithms
	properties
		type;
	end	
	
	methods
		function self = FeatureSelector(type)
			self.type = type;
		end
		
		function [w] = run(self,X,Y)
		    %X: numdim*numsamples data matrix
		    %Y: numsamples*1 label vector
		    %w: numdim*1 weight vector
		    
			switch self.type
			    case 'ttest'
                    if length(unique(Y)) > 2 
                        error('only support binary labels');
                    end			    
			        [out] = FeatureSelector.fsTtest(X',Y);
                    w = out.W;
                case 'fisher'
                    [out] = FeatureSelector.fsFisher(X',Y);
                    w = out.W;
                case 'spectrum'
                    %warning, cannot be used when # of samples are too large
                    [w] = FeatureSelector.fsSpectrum( X'*X, X');
			    otherwise
			        error('type not implemented yet');
			end
		end
		
		
	end	
	
	methods(Static)
        function [out] = fsTtest(X,Y)
            [~,n] = size(X);
            W = zeros(n,1);
            
            for i=1:n
                X1 = X(Y == 1,i);
                X2 = X(Y == 2,i);
        
                n1 = size(X1,1);
                n2 = size(X2,1);
        
                mean_X1 = sum(X1)/n1;
                mean_X2 = sum(X2)/n2 ;   
        
                var_X1 = sum((X1 - mean_X1).^2)/n1;
                var_X2 = sum((X2 - mean_X2).^2)/n2;
                
                W(i) = (mean_X1 - mean_X2)/sqrt(var_X1/n1 + var_X2/n2);
        
            end
            out.W = W;
%            [out.W out.fList] = sort(W, 'descend');    
%            out.prf = 1;
        end
        
        function [out] = fsFisher(X,Y)
            %Fisher Score, use the N var formulation
            %   X, the data, each row is an instance
            %   Y, the label in 1 2 3 ... format
            
            numC = max(Y);
            [~, numF] = size(X);
            out.W = zeros(1,numF);
            
            % statistic for classes
            cIDX = cell(numC,1);
            n_i = zeros(numC,1);
            for j = 1:numC
                cIDX{j} = find(Y(:)==j);
                n_i(j) = length(cIDX{j});
            end
            
            % calculate score for each features
            for i = 1:numF
                temp1 = 0;
                temp2 = 0;
                f_i = X(:,i);
                u_i = mean(f_i);
                
                for j = 1:numC
                    u_cj = mean(f_i(cIDX{j}));
                    var_cj = var(f_i(cIDX{j}),1);
                    temp1 = temp1 + n_i(j) * (u_cj-u_i)^2;
                    temp2 = temp2 + n_i(j) * var_cj;
                end
                
                if temp1 == 0
                    out.W(i) = 0;
                else
                    if temp2 == 0
                        out.W(i) = 100; %note that here's a constant here
                    else
                        out.W(i) = temp1/temp2;
                    end
                end
            end            
        end
        
        function [ wFeat, SF ] = fsSpectrum( W, X, style, spec )
            %function [ wFeat, SF ] = fsSpectrum( W, X, style, spec )
            %   Select feature using the spectrum information of the graph laplacian
            %   W - the similarity matrix or a kernel matrix
            %   X - the input data, each row is an instance
            %   style - -1, use all, 0, use all except the 1st. k, use first k except 1st.
            %   spec - the spectral function to modify the eigen values.
            
            [numD,numF] = size(X);
            
            if nargin < 4 || ~isa(spec, 'function_handle')
                spec = @(X)(X);
            end
            
            % build the degree matrix
            D = diag(sum(W,2));
            % build the laplacian matrix
            L = D - W;
            
            % D1 = D^(-0.5)
            d1 = (sum(W,2)).^(-0.5);
            d1(isinf(d1)) = 0;
            
            % D2 = D^(0.5)
            d2 = (sum(W,2)).^0.5;
            v = diag(d2)*ones(numD,1);
            v = v/norm(v);
            
            %  build the normalized laplacian matrix hatW = diag(d1)*W*diag(d1)
            hatL = repmat(d1,1,numD).*L.*repmat(d1',numD,1);
            
            % calculate and construct spectral information
            [V, EVA] = svd(hatL,'econ');
            EVA = diag(EVA);
            EVA = spec(EVA);
            
            % begin to select features
            wFeat = ones(numF,1)*1000;
            
            for i = 1:numF
                f = X(:,i);
                hatF = diag(d2)*f;
                l = norm(hatF);
            
                if l < 100*eps
                    wFeat(i) = 1000;
                    continue;
                else
                    hatF = hatF/l;
                end
            
                a = hatF'*V;
                a = a.*a;
                a = a';
            
                switch style
                    case -1, % use f'Lf formulation
                        wFeat(i) = sum(a.*EVA);
                    case 0, % using all eigenvalues except the 1st
                        a(numD) = [];
                        wFeat(i) = sum(a.*EVA(1:numD-1))/(1-(hatF'*v)^2);
                    otherwise,
                        a(numD) = [];
                        a(1:numD-style) = [];
                        wFeat(i) = sum(a.*(2-EVA(numD-style+1:numD-1)));
                end
            end
            
            SF = 1:numD;
            
            if style ~= -1 && style ~= 0
                wFeat(wFeat==1000) = -1000;
            end                        
        end
	end
end