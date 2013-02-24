classdef SparseFiltering < handle & Learner
    %sparse filtering wrapper
    properties                
        weights;       %feadim*numunits                
        numunits;
        
        type; %activation type
        
        DC_remove = true;        
        max_iter_per_batch = 300;
        batch_size;
        
        %temp
        Obj;
        OUT; %numunits * numdata                
    end
    
    methods
        function self = SparseFiltering(N, type)            
            addpath(genpath(fullfile(Config.lib_dir_path,'minFunc')));            
            
            self.numunits = N;
            self.type = type;            
        end
        
        function train(self, X)        
            numsamples = size(X,2);
            if isempty(self.batch_size)
                self.batch_size = numsamples;            
            end
            
            if self.DC_remove
                disp('remove DC component per samples');
                X = bsxfun(@minus, X, mean(X));
            end
            			
            if isempty(self.weights)
                optW = randn(self.numunits, size(X, 1));
                self.weights = optW';				
            else
                optW = self.weights';
            end            
            
            obj_history = [];
            for t = 1 : self.max_iter                
                randidx = randperm(numsamples);                                
                obj_epoch = [];
                for b=1:floor(numsamples/self.batch_size)
                    batch_idx = randidx((b-1)*self.batch_size+1:min(b*self.batch_size, numsamples));                
                    optW = minFunc(@self.fobj, optW(:), struct('MaxIter', self.max_iter_per_batch), X(:,batch_idx));
                    obj_epoch = [obj_epoch, self.Obj];
                end

                optW = reshape(optW, [self.numunits, size(X, 1)]);    
                self.weights = optW';
                obj_history = [obj_history mean(obj_epoch)];

%               %save if needed
                if nnz(self.save_iter == t) > 0
                    self.save();
                end
            end                                                           
        end
        
        function [Obj, DeltaW] = fobj (self, paramvec,X)
            % Reshape W into matrix form			
            W = reshape(paramvec, [self.numunits, size(X,1)]);
            % Feed Forward
            F = W*X; % Linear Activation %numunits*numsample
            switch self.type
                case 'soft_abs'
                    Fs = sqrt(F.^2 + 1e-8); % Soft-Absolute Activation
                case 'sig'    
                    Fs = Utils.sigmoid(F);
                case 'rec'
                    Fs = F;
                    Fs(Fs < 0) = 0; 
            end

            [NFs, L2Fs] = SparseFiltering.l2row(Fs); % Normalize by Rows
            [Fhat, L2Fn] = SparseFiltering.l2row(NFs'); % Normalize by Columns
            % Compute Objective Function
            Obj = sum(sum(Fhat, 2), 1);
            self.Obj = Obj;
            % Backprop through each feedforward step
            DeltaW = SparseFiltering.l2grad(NFs', Fhat, L2Fn, ones(size(Fhat)));
            DeltaW = SparseFiltering.l2grad(Fs, NFs, L2Fs, DeltaW');
            switch self.type
                case 'soft_abs'
                    DeltaW = (DeltaW .* (F ./ Fs)) * X';
                case 'sig'
                    DeltaW = (DeltaW .* (1-Fs).* Fs) * X';
                case 'rec'
                    DeltaW = (DeltaW .* (F>0)) * X';              
            end
            DeltaW = DeltaW(:);
        end
        
        function checkGradient(self)
            feadim = 3;
            numunits = 2;
            numsamples = 4;
                       
            X = rand(feadim,numsamples);
            sf_gc = SparseFiltering(numunits, self.type);
            x0 = randn(numunits, feadim);
            
            [d dbp dnum] = Utils.checkgrad2_nodisplay(@(paramvec) sf_gc.fobj(paramvec, X), x0, 1e-5);                       
            fprintf('diff=%g, norm(dbp-dnum)/norm(dbp)=%g\n', d, norm(Utils.vec(dbp-dnum))/(1e-8+norm(Utils.vec(dbp))));
        end
        
        function [acti] = fprop(self, X)
            % Feed Forward
%             disp('remove DC component per samples');
            if self.DC_remove
                X = bsxfun(@minus, X, mean(X));
            end
            
            F = self.weights'*X; % Linear Activation
            switch self.type
                case 'soft_abs'
                    Fs = sqrt(F.^2 + 1e-8); % Soft-Absolute Activation
                case 'sig'    
                    Fs = sigmoid(F);
                case 'rec'
                    Fs = F;
                    Fs(Fs < 0) = 0; 
            end
                        
            [NFs, ~] = SparseFiltering.l2row(Fs); % Normalize by Rows
            [self.OUT, ~] = SparseFiltering.l2row(NFs'); % Normalize by Columns
            self.OUT = self.OUT';
            acti = self.OUT;                        
        end

        
        function [] = save(self) 
            %should add more items to learner_id in the future
            learner_id = sprintf('SparseFiltering_nu%d',self.numunits);
            savedir = fullfile(Config.basis_dir_path,self.save_dir);                       
            if ~exist(savedir,'dir')               
                mkdir(savedir);
            end
            savepath = fullfile(savedir, learner_id);
            
            learner = self;
            save([savepath '.mat'],'learner');
            
            clf;
            Utils.display_network_l1(self.weights);
            saveas(gcf,[savepath '_1.png']);            
        end
        
        % function [stats, hprobs] = get_statistics(self, X)          
            % numdata = size(X,2);
            % self.fprop(X);
            % hprobs = self.OUT;
            
            % stats.sparsity = mean(mean(hprobs));
        % end
        

        
    end
    methods(Static)
        function [G] = l2grad(X,Y,N,D) % Backpropagate through Normalization
            G = bsxfun(@rdivide, D, N) - bsxfun(@times, Y, sum(D.*X, 2) ./ (N.^2));
        end
        
        function [Y,N] = l2row(X) % L2 Normalize X by rows
            % We also use this to normalize by column with l2row(X')
            N = sqrt(sum(X.^2,2) + 1e-8);
            Y = bsxfun(@rdivide,X,N);
        end
        
        %----------testing script----------------
        function [] = test_natural_image()		
            %a little noisy but does result gabor filers
			disp('test with natural image')
			data = DataLoader();
			data.loadNaturalImage();
			            
            numunits = 100;
            type = 'soft_abs';
            clear sf;
            sf = SparseFiltering(numunits,type);	
            sf.max_iter = 1;
            sf.save_iter = 1:sf.max_iter;
            sf.save_dir = 'test';
            
			sf.train(data.Xtrain(:,1:20000));
		end 
    end
end