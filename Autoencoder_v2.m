classdef Autoencoder_v2 < handle & Learner
%Autoencoder that apply OO        
    properties        
        nn; %neural network        
		opt; %optimizer
        dbn; %for easiness to run commands (should be called by reference)
		
        lower_learner; %weights in lower layer, only for visualization
        vbias;    
                
        type; %1:sigmoid-linear + sqaured error(gau), 2: sigmoid-sigmoid + cross-entropy (bin) 
           
        contractive_C = 0;;        
        target_sparsity = 0;
        lambda_sparsity = 0;
        noise = 0;                                
        
        batch_size = 1000;        
                          
        obj_history;
        error_history;
        sparsity_history;
        				
        %temporary 
        numParam;        
        mask;        
        l2_C_tmp;        
        recon_err;
        sparsity;
        J;
        Obj;
        hprobs;                  
    end
    
    methods        
        function self = Autoencoder_v2(numunits, type)                                                                                  
            self.opt = Optimizer();
            self.nn = NeuralNetwork(numunits);			
			self.dbn = DeepBeliefNetwork();
			self.dbn.add(self.nn);			
            self.type = type;           
        end
                        
        function train(self, X)     
            %initialization                        
            %put this into Learner's function?                        
            if isempty(self.obj_history)                                             
                tstart = 0;
            else
                disp('use existing weights');
                tstart = length(self.error_history);
                self.save_iter = self.save_iter + tstart;
            end
            self.setPar(size(X));   
            
            
            if isempty(self.save_iter)
                self.save_iter = self.max_iter;
            end
                        
            
            %keep this?
            cae_history = []; 
            
            for t= tstart+1 : tstart+self.max_iter                                                                                
                randidx = randperm(size(X,2));                
                recon_err_epoch = [];
                sparsity_epoch = [];  
                cae_epoch = [];
                obj_epoch = [];               
                
                for b=1:floor(size(X,2)/self.batch_size)
                    tic
                    batch_idx = randidx((b-1)*self.batch_size+1:min(b*self.batch_size, size(X,2)));                
                    theta = self.vectorizeParam();                    
                    %considering running avg. afterwards.
%                     if self.lambda_sparsity ~= 0 
%                         self.target_sparsity_tmp  =  self.target_sparsity;
%                     end
                                                                
                    if self.noise ~= 0 
                        self.mask = (Utils.rand([self.nn.in_size, self.batch_size]) > self.noise);                         
                    end
                    
                    theta = self.opt.run(@(paramvec) self.fobj(paramvec, X(:,batch_idx)), theta);                                       
                    self.devectorizeParam(theta);                    
                    
                    recon_err_epoch = [recon_err_epoch self.recon_err];
                    sparsity_epoch = [sparsity_epoch self.sparsity];
                    cae_epoch = [cae_epoch self.J];
                    obj_epoch = [obj_epoch self.Obj];
                    
%                     fprintf('%g, %g, %g\n', self.recon_err, self.J, self.sparsity);
                    toc
                end
                self.error_history = [self.error_history mean(recon_err_epoch)];
                self.sparsity_history = [self.sparsity_history mean(sparsity_epoch)];
                self.obj_history = [self.obj_history mean(obj_epoch)];                                
                
                cae_history = [cae_history mean(cae_epoch)];                
                
                %save if needed
                if nnz(self.save_iter == t) > 0
                    self.save();
                end
            end            
        end
        
        function [f g] = fobj(self, paramvec, X)                       
            self.devectorizeParam(paramvec);            
            
            if self.noise ~= 0
                self.dbn.fprop(self.mask.*X)
            else                
                self.dbn.fprop(X);
            end
			
			%reverse pooling                        
            recons = self.nn.weights*self.nn.OUT + repmat(self.vbias,[1 self.batch_size]);                        
					
			if self.type == 'bin'
				recons = Utils.sigmoid(recons);
			end
			
            delta = recons-X;
            if self.type == 'gau'
                dyhat = 2*delta;
                f = sum(delta(:).^2);
            elseif self.type == 'bin'
                dyhat = delta;
                f = -sum(Utils.vec(X.*log(recons+1e-8) + (1-X).*log(1-recons+1e-8)));
            end
            
			self.recon_err = f/self.nn.numdata;
            self.sparsity = mean(self.nn.OUT(:)); %can still track saturation rate
			
            dWdec = dyhat*self.nn.OUT';
			dvbias = sum(dyhat,2);
            dhprobs = self.nn.weights'*dyhat;
                        
            f = f + self.dbn.bprop(dhprobs);           
            g = self.dbn.vectorizeParamGrad();
			g(1:numel(self.nn.weights)) = g(1:numel(self.nn.weights)) + dWdec(:); %tied weights
            g = [g; dvbias];
            
            %---------------- need clean up -----------                                                 
                                       
            if self.contractive_C ~= 0
                error('not implemented yet')
                h2 = ha.^2; %sum over data
                one_minus_twoh = (1-2*hprobs);
                w2 = sum(self.weights.^2,1);
                J = sum(sum(h2,2).*w2');
                self.J = J/self.batch_size;
                f = f + 0.5 * self.contractive_C * J;  %already consider batch_size     
                
                dhbias = dhbias + self.contractive_C * w2' .* sum(h2 .* one_minus_twoh,2);                
                dW = dW + self.contractive_C * ( bsxfun(@times,X * (h2.*one_minus_twoh)' , w2) + bsxfun(@times,self.weights, sum(h2,2)')) ;
            end                       
            
            self.Obj = f;                                        
        end        
                
        function theta = vectorizeParam(self)
                theta = [self.dbn.vectorizeParam() ; self.vbias];
        end
        
        function [] = devectorizeParam(self, paramvec)
            self.dbn.devectorizeParam(paramvec(1:self.nn.paramNum));
            self.vbias = paramvec(self.nn.paramNum+1:end);
        end
        
        function setPar(self, datadim)            
            numsamples = datadim(end);  
            if numsamples < self.batch_size
                self.batch_size = numsamples;
            end
            
            self.dbn.setPar([datadim(1:end-1) self.batch_size]);
            if isempty(self.vbias)
                self.vbias = Utils.zeros([datadim(1),1]);
            end
        end               
        
        function checkGradient(self)
            numunits= 2;			
			feadim = 6;
			numdata = 3;			
			
            ae = Autoencoder_v2(numunits, self.type);
            for i = 2:self.dbn.nlayer
                ae.dbn.add(self.dbn.nnet{i});
            end
			X = rand(feadim,numdata);
			ae.setPar(size(X));      
			
            
            %self.batch_size = numsamples;
            %self.l2_C_tmp = self.l2_C*numsamples;
            %self.weights = 1e-3*randn(self.feadim,self.numunits);
            %self.hbias = zeros(self.numunits,1);
            %self.vbias = zeros(self.feadim,1);               
            %self.numParam = [numel(self.weights), numel(self.hbias), numel(self.vbias)];                            
            %self.numParam = cumsum(self.numParam);
            x0 = ae.vectorizeParam();
            
            %if self.noise ~= 0
            %    self.mask = (rand(size(X)) > self.noise);
            %end
            
            [d dbp dnum] = Utils.checkgrad2_nodisplay(@(paramvec) ae.fobj(paramvec, X), x0, 1e-5);            
            fprintf('diff=%g, norm(dbp-dnum)/norm(dbp)=%g\n', d, norm(Utils.vec(dbp-dnum))/(1e-8+norm(Utils.vec(dbp)))); 
        end                        
        
        function [hprobs] = fprop(self,X)            
            %TODO
        end
      
            
        function [] = save(self) 
             %should add more items to learner_id in the future
            learner_id = sprintf('Autoencoder_nu%d_l2%g_sp%g_spl%g',self.nn.numunits, self.nn.l2_C, self.target_sparsity, self.lambda_sparsity);
            savedir = fullfile(Config.basis_dir_path,self.save_dir);                       
            if ~exist(savedir,'dir')               
                mkdir(savedir);
            end
            savepath = fullfile(savedir, learner_id);
            
            learner = self;
            save([savepath '.mat'],'learner');
                        
            figure(1);
            Visualizer.display_network_l1(self.nn.weights); %TODO: make this be able to display upper layer
            saveas(gcf,[savepath '_1.png']);
            
            %currently only show obj, error & sparsity            
            figure(2);
            subplot(3,1,1);
            plot(self.obj_history);
            subplot(3,1,2);
            plot(self.error_history);
            subplot(3,1,3);
            plot(self.sparsity_history);
            saveas(gcf,[savepath '_2.png']);
        end
        
    end
       
    methods(Static)
		function [] = test_natural_image()			
            %WARNING: still cannot get gabor filters, is it the problem of LBFGS?
			disp('test with natural image')
			data = DataLoader();
			data.loadNaturalImage();
			            
            numunits = 128;
            type = 'gau';
            clear ae;
            
            km = Kmeans(numunits,'tri');
            km.train(data.Xtrain);
            
            ae = Autoencoder_v2(numunits, type);
%             ae.noise = 0.5;
            gs = GroupSparsity(1,1);
            gs.lambda = 1;
            gs.target_sparsity = 0.02;
            gs.pass_up = false;
%             ae.dbn.add(gs);
            
            ae.nn.l2_C = 1;
            ae.nn.init_weight = 0.1;
            ae.nn.weights = 0.1*km.weights;%initialized from random patch
            ae.nn.biases = Utils.zeros([numunits,1]);
            
            ae.batch_size = 100;
            ae.max_iter = 20;
            ae.save_iter = 1:20;
            ae.save_dir = 'test';            
%             ae.opt.par = struct('maxFunEvals',100);
            ae.opt = SGD();
%             ae.opt.momentum_par = 1600*5; %when to change momentum
            ae.opt.init_eps = 1e-4;
            ae.opt.eps_par(1) = 0.9998;
            ae.opt.init_momentum=0;
            ae.train(data.Xtrain);
		end
	end	   
       
end