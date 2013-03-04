classdef DataProcessor < handle
%class for data preprocessing
    properties
    end
    
    methods
    end
    
    methods(Static)
		function [Xout] = spectrogram(X, frequency, freq_span)			
			num_frame_for_fft = size(X,1);
			freq_in_hertz = frequency/2 * linspace(0,1,ceil(num_frame_for_fft/2)); 
			freq_chosen_idx = [(freq_in_hertz >= freq_span(1))&(freq_in_hertz <= freq_span(2))];
			
			Xout = zeros(nnz(freq_chosen_idx),size(X,2));
			for i = 1 : size(X,2)
				tmp = abs(fft(X(:,i)));				
				tmp = tmp(1:ceil(num_frame_for_fft/2));     
				Xout(:,i) = tmp(freq_chosen_idx);
			end						
		end		
	
        function [Xout, Yout] = concatenate(X, step_past, step_future, Y)
            % X: feadim*numsamples data matrix
            % step_past/step_future: binary vector indicating which neighboring frame should be concatenated
            % Y : numsamples*1 label vector
            
            %tool functions (what's the best way to share?)
            vec = @(x)x(:);
        
            %X should be feadim*numsamples
            [feadim numsamples] = size(X);
            window_size = (length(step_past)+length(step_future)+1);
            feature_size = nnz(step_past)+nnz(step_future)+1;
            Xout = zeros(feadim*feature_size,numsamples-length(step_past)-length(step_future));
            
            sliding_window = [1:window_size];
            sliding_window = sliding_window([step_past 1 step_future]==1);
            
            for i = 1 : size(Xout,2)                              
                Xout(:,i) = vec(X(:,sliding_window));
                sliding_window = sliding_window+1;
            end

            %also process labels if exist        
            if exist('Y','var')
                Yout = Y(length(step_past)+1:length(Y)-length(step_future));
            end
        end
        
        function [Xtrain, Xtest, M, S] = normalize(Xtrain, Xtest, M, S)
            if ~exist('M', 'var') || isempty(M)
                M = mean(Xtrain,2);
                S = std(Xtrain, [], 2)+1e-5;
            end
            Xtrain = bsxfun(@rdivide, bsxfun(@minus, Xtrain, M), S);
            if exist('Xtest','var') && ~isempty(Xtest)
                Xtest = bsxfun(@rdivide, bsxfun(@minus, Xtest, M), S);
            end
        end
	
        function [X_zca M P numfactor] = ZCA_whitening(X, numfactor, reg)
             [~, M ,P ,numfactor, U] = DataProcessor.PCA_whitening(X, numfactor, reg);
             P = U(:,1:numfactor)*P;
             X_zca = P*X;
        end
		
		function [X_pca M P numfactor U] = PCA_whitening(X, numfactor, reg)			
            if ~exist('reg', 'var') reg = 0; end
            
            M = mean(X,2);
            X = bsxfun(@minus, X, M);   
            
            [U, S] = svd(X*X'/size(X,2));
            [s ,idx] = sort(diag(S), 'descend');  
            U = U(:,idx);

            loading = cumsum(s) / sum(s);
            
            if numfactor <= 1
                numfactor = nnz(loading <= numfactor);
                fprintf('number of pc: %g\n', numfactor);                
            else
                fprintf('percentage of pc: %g\n', loading(numfactor));
            end
            
            P = bsxfun(@times,U(:,1:numfactor), 1./sqrt(s(1:numfactor)'+reg))'; %number of pc * feadim            
			X_pca = P*X;
        end
		
		function y = rescale(x,a,b)
			% rescale - rescale data in [a,b]						
			if nargin<2
				a = 0;
			end
			if nargin<3
				b = 1;
			end

			m = min(x(:));
			M = max(x(:));

			if M-m<eps
				y = x;
			else
				y = (b-a) * (x-m)/(M-m) + a;
			end
        end
        
        function patches = random_patches(X, rfSize, numPatches) %anothers version of getdata_imagearray from Adam's kmeans paper
            %X = numsamples * feadim
            [M N numchannels numdata]= size(X);
            patches = Utils.zeros([rfSize*rfSize*numchannels,numPatches]);
            for i=1:numPatches
              if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, numPatches); end

              r = random('unid', M - rfSize + 1);
              c = random('unid', N - rfSize + 1);              
              patch = X(r:r+rfSize-1,c:c+rfSize-1,:,mod(i-1,numdata)+1);              
              patches(:,i) = patch(:);
            end            
        end
        
        %text related
        
        %misc
        function [Xout] = cell2num(X)
            %transform cell array to numbers, all strings are simply treated as NaN
            Xout = zeros(size(X));
            for i = 1 : numel(X)
                if isnumeric(X{i})
                    Xout(i) = X{i};
                    continue;
                end
                
                tmp = str2double(X{i});
                if ~isempty(tmp)
                    Xout(i) = tmp;
                else
                    Xout(i) = NaN;
                end
            end
        end
        
    end
end