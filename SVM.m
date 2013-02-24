classdef SVM < handle & Classifier
%the wrapper for many different SVM libraries
%only linear_minFunc are updated to latest version, other types might need debug and update
    properties
        model;
		par;
        type; %linear_minFunc, liblinear, libsvm, svmlight, svmperf		
    end
    
    methods
        function self = SVM(type, par)                        
            self.type = type;
			if strcmp(self.type, 'liblinear')
				addpath /mnt/neocortex/library/liblinear-1.8/matlab;  	
			elseif strcmp(self.type, 'libsvm')
				addpath(fullfile(Config.lib_dir_path,'libsvm-3.12/matlab'));
            elseif strcmp(self.type, 'svmlight')
                addpath(genpath(fullfile(Config.lib_dir_path,'svm_light')));
            elseif strcmp(self.type, 'linear_minFunc')
                %linear SVM without bias, optimized by minFunc, Y can only be positive integers
				addpath(genpath(fullfile(Config.lib_dir_path,'minFunc')));                                 					
            elseif strcmp(self.type, 'svmperf')                
                addpath(fullfile(Config.lib_dir_path,'svmperf'));
			end
             
			if exist('par','var') 	
				self.par = par;
			end			
        end
                
        function train(self,X,Y)                        
            switch self.type
                case 'svmperf'
                    %http://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html					
					ylist = unique(Y);
					if length(ylist > 2) %one vs. all
						self.model = [];
						for i = 1 : length(ylist)
							ytmp = -1*ones(length(Y),1);
							ytmp(Y == ylist(i)) = 1;
							self.model = [self.model svmperflearn(X',ytmp,self.par)];
						end
						self.model(1).ylist = ylist;
					else
						self.model.ylist = ylist;
						ytmp = Y;
						ytmp(Y == ylist(1)) == -1;
						ytmp(Y == ylist(2)) == 1;
						self.model = svmperflearn(X',ytmp,self.par);
					end
                case 'liblinear'            
                    self.model = train(Y, sparse(X), sprintf('-s 0 -c %g  -q', self.C), 'col'); 
				case 'libsvm'				
					self.model = svmtrain(Y, X', ['-s 0 -t 2 ' self.C]);
                case 'svmlight'
                    if length(unique(Y)) > 2
                        error('svmlight only supports binary classification');
                    end
                    Y(Y~=1) = -1;                    
                    self.model = svmlearn(X', Y(:), ['-t 2 -v 2 '  self.C]); 
                case 'linear_minFunc'				
				  numClasses = max(Y); 
				  numsamples = size(X,2);
				  % X = [X' ones(numsamples,1)]; %for bias
				  X = X';
				  w0 = randn(size(X,2)*numClasses, 1); 				  
				  if isfield(self.par,'minFunc')
					minFunc_par = self.par.minFunc;
				  else
					minFunc_par = [];
				  end
				  w = minFunc(@self.my_l2svmloss, w0, minFunc_par, X, Y, numClasses, self.par.C);
				  self.model.weights = reshape(w, size(X,2), numClasses);   %include bias                 				  				  
            end            
        end
        
        function [loss, g] = my_l2svmloss(self, w, X, y, K, C)
              [M,N] = size(X); %numdata*dim
              theta = reshape(w, N,K);
              Y = bsxfun(@(y,ypos) 2*(y==ypos)-1, y, 1:K); % numdata*numcase

              margin = max(0, 1 - Y .* (X*theta)); %numsamples * numclass			  
			  if ~isempty(self.class_penalty)
				 margin = bsxfun(@times, margin, self.class_penalty');
			  end
			  if ~isempty(self.sample_penalty)			  
				 margin = bsxfun(@times, margin, self.sample_penalty);
			  end
			  
              loss = (0.5 * sum(theta.^2)) + C*mean(margin.^2);
              loss = sum(loss);  
              g = theta - 2*C/M * (X' * (margin .* Y));
              g = g(:);
        end 
         
        function [pred accu] = classify(self, X, Y, options) %options = options only for classify
            if ~exist('options','var') || isempty(options)
                options = [];
            end
			if ~exist('Y','var') || isempty(Y)
				Y = ones(size(X,2),1);
			end
            confidence = [];
            switch self.type
                case 'svmperf'
					if length(self.model) > 1 %multi-class
						confidence = zeros(length(Y),length(self.model));
						for i = 1 : length(self.model)							
							confidence(:,i) = svmperfclassify(X',ones(length(Y),1),self.model(i),options);
						end
						[~, idx] = max(confidence, [], 2);
						pred = zeros(length(Y),1);
						for i = 1 : length(self.model)
							pred(idx == i) = self.model(1).ylist(i);
						end
					else
						tmpY = -1*ones(length(Y),1);
						tmpY(Y==self.model.ylist(2)) = 1;
						predictions = svmperfclassify(X',tmpY,self.model,options);
						pred = ones(length(Y),1)*self.model.ylist(2);
						pred(predictions >= 0) = self.model.ylist(1);
					end                                                            
                case 'liblinear'
                    [pred, accu] = predict(Y, sparse(X), self.model, [],'col');    
				case 'libsvm'
					[pred, accu confidence] = svmpredict(Y, X', self.model, '-b 0');
					accu = accu(1)*0.01;
                    
                    % confidence = (confidence(:,1) - confidence(:,2))';
                    confidence = [];
                case 'svmlight'
                    if length(unique(Y)) > 2
                        error('svmlight only supports binary classification');
                    end
                    oriY = Y(Y~=1);
                    Y(Y~=1) = -1;
                    
                    [err confidence] = svmclassify(X', Y(:),self.model);
                    pred = confidence > 0;                    
                    if ~isempty(oriY) && ~isempty(pred==0)
                        pred(pred==0) = oriY(1);
                    end
                    accu = 1-err;
                    fprintf('accu = %g', accu);
                case 'linear_minFunc'                    
					% X = [X; ones(1,size(X,2))];
                    WX = self.model.weights'*X;
                    [val, pred] = max(WX, [], 1);  										
            end
			accu = [];
			pred = pred(:);
			if exist('Y','var')
				accu = mean(pred == Y);                                                            
			end
        end                        
        
        function [accu_train, accu_test, svm] = tuning(self, C, Xtrain, Ytrain, Xtest, Ytest)
            %tuning (Xtest, Ytest are actually validation)
	    if ~isempty(self.Data)
		if exist('Xtrain','var') && ~isempty(Xtrain)	disp('using existing data'); end
		Xtrain = self.Data.Xtrain; Ytrain = self.Data.Ytrain; Xtest = self.Data.Xtest; Ytest = self.Data.Ytest;
	    end
	    
	    if ~isempty(self.Clist)
		if exist('C','var') && ~isempty(C) disp('using existing Clist'); end
		C = self.Clist;
	    end
			    
            num_C = size(C,1);
            accu_train = zeros(num_C,size(C,2)+1);
            accu_test = zeros(num_C,size(C,2)+1);
			
			%better way for this?
            accu_train(:,1:size(C,2)) = C;
            accu_test(:,1:size(C,2)) = C;
            
            for i = 1 : length(C)
                self.C = C(i,:);
                self.train(Xtrain,Ytrain);
                [~, accu] = self.classify(Xtrain,Ytrain);
                accu_train(i,end) = accu;

                [~, accu] = self.classify(Xtest,Ytest);
                accu_test(i,end) = accu;
            end
            
            if nargout > 2
                [~, idx] = max(accu_test(:,end));
                self.C = accu_test(idx,1:size(C,2));
                self.train([Xtrain Xtest], [Ytrain;Ytest]);
                svm = self;
            end
        end
        
        function [accu_train, accu_cv, accu_test] = cross_validation(self, Xtrain, Ytrain, Xtest, Ytest, C, K)
            idx = randperm(length(Ytrain));
            Xtrain = Xtrain(:, idx);
            Ytrain = Ytrain(idx);
            
            batch_size = floor(length(Ytrain) / K);
            accu = zeros(K, length(C));
            for i = 1 : K 
                cv_idx = (i-1)*batch_size + 1  : i*batch_size;
                for j =  1: length(C)
                     self.C = C(j);
                     xcurr = Xtrain;
                     ycurr = Ytrain;
                     xcurr(:,cv_idx) = [];
                     ycurr(cv_idx) = [];
                     self.train(xcurr,ycurr);
                     [~, accu(i,j)] = self.classify(Xtrain(:,cv_idx), Ytrain(cv_idx));
                end
            end
            
            mean_accu = mean(accu,1);
            [accu_cv, best_C_idx] = max(mean_accu);
            self.C = C(best_C_idx);
            
            self.train(Xtrain, Ytrain);
            [~, accu_train] = self.classify(Xtrain, Ytrain);
            [~, accu_test] = self.classify(Xtest, Ytest);
        end
        
    end
    
   
end