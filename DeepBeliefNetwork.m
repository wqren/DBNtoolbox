classdef DeepBeliefNetwork < handle & Classifier
%
%the second version of CNN toolbox, currently supports less functions and is simpler
%
    properties
        nnet;        
        nlayer;        
        opt; %optimizer
        
        %training parameters
        reset_before_train = false; %will reset parameters each time running train()

        batch_size;
        max_iter = 1;   %default setting is using lbfgs, update with all data        
    end
    
    methods
        function self = DeepBeliefNetwork()     
            self.nlayer = 0;
            self.nnet = {};
            self.opt = Optimizer();
        end
                
        
        function add(self, obj)
            %TODO : check validality
            self.nnet{self.nlayer+1} = obj;
            self.nlayer = self.nlayer +1;
            if self.nlayer == 1
                self.nnet{1}.skip_passdown = true;
            end
        end
        
        function reset(self)
            for i = 1 : length(self.nnet)
                self.nnet{i}.reset();
            end
        end
        
		function removeTop(self)
			assert(self.nlayer > 0);
		    self.nnet = self.nnet(1:self.nlayer-1);
			self.nlayer = self.nlayer - 1;
		end
		
        function skip_layer(self, nlayer)
            if nlayer > self.nlayer
                error('nlayer not valid');
            end
            self.nnet{nlayer}.skip_update = true;
            self.nnet{nlayer}.paramNum = 0;
        end
        
        function [out] = fprop(self, X, nlayer) % This does not perform backpropagation
            if ~exist('nlayer', 'var')
                nlayer = self.nlayer;
            end
            assert(self.nlayer > 0 && nlayer <= self.nlayer);                       
            
            %below should be pass by reference under MATLAB compiler, but should be checked with profiler
            self.nnet{1}.IN = X;
            for i=1:nlayer - 1                
                self.nnet{i}.fprop();
                self.nnet{i+1}.IN = self.nnet{i}.OUT;                                       
            end                            
            self.nnet{nlayer}.fprop();
            
            if nargout > 0
                out = self.nnet{nlayer}.OUT;
            end
        end        

        
        function [f, derivative] = bprop(self,y)   
            %self.fprop(X);  not include forward anymore
            [f derivative] = self.nnet{self.nlayer}.bprop(0, y);    
            for i=self.nlayer-1:-1:1               
                [f derivative] = self.nnet{i}.bprop(f,derivative);  
            end  
        end

        
        function clearGradient(self) % cleans up all intermediate variables and gradients, but not the parameters
            for i = 1 : self.nlayer
                self.nnet{i}.IN = [];
                self.nnet{i}.OUT = [];
            end
        end
        
        function [f g] = fobj(self, paramvec, X, y)            
            %should check will vectorize and devectorize take lots of time or not
            self.devectorizeParam(paramvec);
            self.fprop(X);
            f = self.bprop(y);
            g = self.vectorizeParamGrad;
        end               
        
        
        
        function train(self, X, y)                        			
			if size(y,2) == 1
				numclass = length(unique(y(:)));
				y_multi = Utils.num2bin(y,numclass); 
			else
				numclass = size(y,1);
                y_multi = y;
			end
            
            %---set parameters---            
            size_x = size(X);
            numsamples = size_x(end);
            if isempty(self.nnet{1}.in_size) %need initialization
                self.setPar(size(X),numclass);            
            end                        
            %---------------------            
            
            %is it needed? need to think about reset issue afterwards
            if self.reset_before_train
                self.reset();
            end            
                
            x = self.vectorizeParam;                        
            
            if ~isempty(self.class_penalty)
                self.classPenaltyOnSamples(y);
            end 
            
            if isempty(self.batch_size)
                batch_size = numsamples;
            else
                batch_size = self.batch_size;
            end            

            for t = 1 : self.max_iter
                randidx = randperm(numsamples);
                for b =  1 : ceil(numsamples/batch_size)
                    batch_idx = randidx((b-1)*batch_size+1 : min(b*batch_size,numsamples));
                    if ~isempty(self.sample_penalty)
                        self.nnet{end}.sample_penalty = self.sample_penalty(batch_idx); 
                    end
                    
                    %training_par should be put in Optimizer in the future
                    switch length(size_x)%for different input format
                        case 2
                            x = self.opt.run(@(paramvec) self.fobj(paramvec, X(:,batch_idx), y_multi(:,batch_idx)), x);                
                        case 4
                            x = self.opt.run(@(paramvec) self.fobj(paramvec, X(:,:,:,batch_idx), y_multi(:,batch_idx)), x);                
                    end
        
                    self.devectorizeParam(x);            
                end
            end
        end
        

        
        function [d dbp dnum] = checkGradient(self,input_format)
            %generate test data            
            M = 7;
            N = 7;
            numchannels = 3;
            numdata = 4;
            numunits = 3;            
            numclass = 2;
            
            if ~exist('input_format','var')
                 input_format = 2;
            end
            
            switch input_format                
                case 2
                    X = randn(numunits, numdata);
                case 4
                    X = randn(M,N,numchannels,numdata);  
                otherwise
                    error('input format not supported');
            end
            
            self.clearGradient();            
            tempDbn = DeepBeliefNetwork;    
            tempDbn.nlayer = 1;
            %parameter to test : original weights, x
            %f = product of OUT and some rand Weights
            for i = 1 : length(self.nnet)
                tempDbn.nnet{1} = self.nnet{i}.gradCheckObject;                
                tempDbn.nnet{1}.skip_passdown = false;
                tempDbn.setPar(size(X),numclass);                
                
                pW = tempDbn.vectorizeParam;
                numW = length(pW);                
                x0 = [pW; Utils.vec(X)];               
                               
                tempDbn.fprop(X);
                                
                RW = randn(size(tempDbn.nnet{1}.OUT)); %random weights              
                
                [d dbp dnum] = Utils.checkgrad2_nodisplay(@(paramvec) tempDbn.fobj_Grad(paramvec, RW, numW, size(X)), x0, 1e-3);
                fprintf('layer = %g, diff=%g, norm(dbp-dnum)/norm(dbp)=%g\n', i, d, norm(Utils.vec(dbp-dnum))/(1e-8+norm(Utils.vec(dbp))));
                
                X = randn(size(tempDbn.nnet{1}.OUT)); %for next layer
            end
                        
            self.clearGradient();
        end

        %objective function for gradient check
        function [f g] = fobj_Grad(self, paramvec, RW, numW,sizeX)
            pW = paramvec(1:numW);
            pX = paramvec(numW+1:end);            
            
            self.devectorizeParam(pW);        
            self.nnet{1}.IN = reshape(pX,sizeX);
            self.nnet{1}.fprop();            
            
            f = sum(Utils.vec(RW .* self.nnet{1}.OUT));
            [f, dX] = self.nnet{1}.bprop(f,RW);                                    
            gW = self.vectorizeParamGrad;
            gX = double(Utils.vec(dX));
            g = [gW; gX];
        end
        
        function paramvec = vectorizeParam(self)
            paramvecsize = 0;
            for i=1:length(self.nnet)                
                paramvecsize = paramvecsize + self.nnet{i}.paramNum;
            end            
                paramvec = zeros(paramvecsize,1);        
            
            count = 0;
            for i=1:length(self.nnet)
                param = self.nnet{i}.getParam;
                for j = 1 : length(param);
                        item = param{j};
                        paramvec(count+1:count+numel(item)) = double(item(:));  %sacrifice GPU speed for illegal check
                        count = count + numel(item);
                end
            end
        end
        
        function paramvec = vectorizeParamGrad(self)
             paramvecsize = 0;
            for i=1:length(self.nnet)                
                paramvecsize = paramvecsize + self.nnet{i}.paramNum;
            end
                        
                paramvec = zeros(paramvecsize,1);
            
            count = 0;
            for i=1:length(self.nnet)
                param = self.nnet{i}.getGradParam;
                for j = 1 : length(param);
                        item = param{j};
                        paramvec(count+1:count+numel(item)) = double(item(:));
                        count = count + numel(item);
                end
            end
        end
                
        function devectorizeParam(self, paramvec)
            count = 0;
            for i=1:length(self.nnet)
                num = self.nnet{i}.paramNum;
                if num~= 0
                    self.nnet{i}.setParam(paramvec(count+1:count+num));
                    count = count + num;
                end
            end

        end
        
       
        function [pred, acc] = classify(self, X, y)
            % pred_arr: classification prediction 
            self.clearGradient;
            self.fprop(X);    
            [~, pred] = max(self.nnet{end}.OUT,[],1);
            pred = pred(:); 
            acc = [];       
			if exist('y','var')
				acc = mean(pred(:) == y(:));
			end
        end
        
        function [] = setPar(self,datadim, numclass)                
            %for multidim data, make sure that numdata is > 1
            numdata = datadim(end);
            if isa(self.nnet{1},'Classifier') %only 1 layer
                self.nnet{1}.setPar(datadim(1:end-1),numclass);
            else
                self.nnet{1}.setPar(datadim(1:end-1));
            end
            
            for i = 2 : self.nlayer-1
                self.nnet{i}.setPar(self.nnet{i-1}.out_size);
            end
            
            if self.nlayer > 1
                if isa(self.nnet{end},'Classifier')
                    self.nnet{end}.setPar(self.nnet{end-1}.out_size,numclass);
                else
                    self.nnet{end}.setPar(self.nnet{end-1}.out_size);
                end            
            end
            
            self.setNumData(numdata);
        end
        
        function [] = setNumData(self,numdata)
            for i = 1 : self.nlayer
                self.nnet{i}.numdata = numdata;
            end
        end
    end
    
    methods(Static, Access = private)
        

    end
    
    %%%%%%%%%%%%%%%%%%some miscellaneous functions (move to other objects?)%%%%%%%%%%%%%%%%%%%
    methods(Static)       
        function ginfo()
            addpath /usr/local/jacket/engine/            
            ginfo
        end
        
        
        
         function save_progress(learner, fname_mat, fname_png_1, fname_png_2, plots, titles)
			try
				if exist('plots','var');
					learner.save_progress(fname_mat, plots); 
				else
					learner.save_progress(fname_mat); 
				end
			catch  %temporary
				disp('using default save')
				if exist('plots','var');
					save(fname_mat, 'learner', 'plots'); 
				else
					save(fname_mat, 'learner'); 
				end
			end
            clf;            
            
			for i = 1 : learner.Ndisplay
				if ~isempty(learner.E)
					DeepBeliefNetwork.display_network(learner.E*learner.show_basis(i), learner.prev_learner ,[], learner.numchannels);                                                            
				else
					DeepBeliefNetwork.display_network(learner.show_basis(i), learner.prev_learner ,[], learner.numchannels);                                                            
				end			
				saveas(gcf, [fname_png_1 '_' num2str(i) '.png']);
			end
            
            clf;
            if exist('fname_png_2', 'var') && ~isempty(fname_png_2)
                for i = 1 : length(plots)
                    subplot(length(plots),1,i), plot(plots{i}); title(titles{i});
                end
                saveas(gcf, fname_png_2); 
            end
         end
        
        function [Ktrain Ktest] = feature_reducedim_chol(Xtrain, Xtest)
            % Note: This code works only for linear kernels

            numfeat = size(Xtrain,1);
            numtrain = size(Xtrain,2);

            % check if the #feat is larger than #ex (if not, there is no point to run
            % this code)
            assert(numfeat > numtrain);

            assert(numfeat == size(Xtest,1));

            % R = chol(A): R'R = A
            % Ktrain = chol(Xtrain'*Xtrain+10^-5*eye(size(Xtrain,2)));
            Ktrain = chol(Xtrain'*Xtrain + 1e-4*eye(size(Xtrain,2)));

            % Ktrain'*Ktest = Xtrain'*Xtest;
            Ktest = (Ktrain')\(Xtrain'*Xtest);

            % Now you can use Ktrain and Ktest as features
        end
         
%%%%%%%%%%%%%%%%%%%%%%preprocessing functions, moving these to LoadData?%%%%%%%%%%%%%%%%%%%%%%        

		
		function [] = fastfig(X, savepath)
			if ~exist( 'savepath','var') savepath = 'tmp.png'; end
			clf;
			DeepBeliefNetwork.display_network_l1(X);
			saveas(gcf,savepath);
		end
end
end
