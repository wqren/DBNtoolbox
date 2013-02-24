classdef ConvAutoencoder < handle & Learner
%TODO: reset before train?    
%the variants such as denoising, sparse, and contractive AE
%second layer : just apply directly on encoded data, only visualization need to change
    
%Convolutional Auto-encoder based on CNN, more OO
%Current design: it's in the same level as DeepBeliefNets, must have a CNN object
% and Pooling2D object to work (maybe adding LCN later)

%based on ICML2011 "On Optimization Methods for Deep Learning", LBFGS is better for low-dimensions input (compared with CG)
%and suggested number of iterations = 20, minibatch = 1000
    properties        
        cnn; %convolutional neural network
        pool; %Pooling2D
		opt; %optimizer
        dbn; %for easiness to run commands (should be called by reference)
		
        lower_learner; %weights in lower layer, only for visualization
        %vbias;    %not use for now, maybe can be used for boundary effect?
                
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
        dhprobs;                        
    end
    
    methods        
        function self = ConvAutoencoder(numunits, ws, type)               
            self.opt = Optimizer();
            self.cnn = CNN(numunits,ws);
			self.pool = Pooling2D(2,2,'max'); %default setting
			self.dbn = DeepBeliefNetwork();
			self.dbn.add(self.cnn);
			self.dbn.add(self.pool);			
            self.type = type;                                                                                   
        end
                        
        function train(self, X)     
            %initialization  
            %TODO: change this to setPar()
            size_x = size(X);
            numsamples = size_x(end);  
            if numsamples < self.batch_size
                self.batch_size = numsamples;
            end
            
            if isempty(self.obj_history)                                             
                tstart = 0;
            else
                disp('use existing weights');
                tstart = length(self.error_history);
                self.save_iter = self.save_iter + tstart;
            end
            self.dbn.setPar([size_x(1:end-1),self.batch_size]);   
            
            if isempty(self.save_iter)
                self.save_iter = self.max_iter;
            end
                        
            
            %keep this?
            cae_history = []; 
            
            for t= tstart+1 : tstart+self.max_iter                                                                                
                randidx = randperm(numsamples);                
                recon_err_epoch = [];
                sparsity_epoch = [];  
                cae_epoch = [];
                obj_epoch = [];               
                
                for b=1:floor(numsamples/self.batch_size)
                    tic
                    batch_idx = randidx((b-1)*self.batch_size+1:min(b*self.batch_size, numsamples));                
                    theta = self.dbn.vectorizeParam();
                    
                    %considering running avg. afterwards.
%                     if self.lambda_sparsity ~= 0 
%                         self.target_sparsity_tmp  =  self.target_sparsity;
%                     end
                                                                
                    if self.noise ~= 0 
                        self.mask = (Utils.rand([self.cnn.in_size, self.batch_size]) > self.noise);                         
                    end
                    
                    theta = self.opt.run(@(paramvec) self.fobj(paramvec, X(:,:,:,batch_idx)), theta);                                       
                    self.dbn.devectorizeParam(theta);    
                    
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
            self.dbn.devectorizeParam(paramvec);
            if self.noise ~= 0
                self.dbn.fprop(self.mask.*X)
            else                
                self.dbn.fprop(X);
            end
			
			%reverse pooling
            [~,unpool_map] = self.pool.bprop(0,self.dbn.nnet{end}.OUT);           
            
            recons = Utils.convReconstruct(unpool_map, self.cnn.weights);
					
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
            
			self.recon_err = f/self.cnn.numdata;
            self.sparsity = mean(self.cnn.OUT(:)); %can still track saturation rate
			
            dWdec = Utils.convGradient(dyhat,unpool_map);
			
            dhprob = Utils.convInference(dyhat, self.cnn.weights);			
			dhprob = self.pool.fprop_rev(dhprob);
            f = f + self.dbn.bprop(dhprob);           
            g = self.dbn.vectorizeParamGrad();
			g(1:numel(self.cnn.weights)) = g(1:numel(self.cnn.weights)) + dWdec(:); %tied weights
            
            %---------------- need clean up -----------                                    
            if self.lambda_sparsity ~= 0 && self.target_sparsity > 0 
				error('not implemented yet')
				%it's actually selectivity
                curr_sparsity = mean(hprobs, 2);
                tmp1 = self.target_sparsity./curr_sparsity;
                tmp2 = (1-self.target_sparsity)./(1-curr_sparsity);
                
                f = f + self.lambda_sparsity*self.batch_size*(sum(self.target_sparsity*log(tmp1) + (1-self.target_sparsity)*log(tmp2)));
                sparsityDelta = self.lambda_sparsity*(-tmp1+tmp2);
                dhprobs = dhprobs + repmat(sparsityDelta,1,self.batch_size);
            end            
                                    
   
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
                 
        
        function checkGradient(self)
			numunits= 2;
			ws =3;
			M = 6;
			N = 6;
			numdata = 3;
			numchannels = 2;
			
            cvae = ConvAutoencoder(numunits, ws, self.type);
            
			X = rand(M,N,numchannels, numdata);
			cvae.dbn.setPar(size(X));      
			
            
            %self.batch_size = numsamples;
            %self.l2_C_tmp = self.l2_C*numsamples;
            %self.weights = 1e-3*randn(self.feadim,self.numunits);
            %self.hbias = zeros(self.numunits,1);
            %self.vbias = zeros(self.feadim,1);               
            %self.numParam = [numel(self.weights), numel(self.hbias), numel(self.vbias)];                            
            %self.numParam = cumsum(self.numParam);
            x0 = cvae.dbn.vectorizeParam();
            
            %if self.noise ~= 0
            %    self.mask = (rand(size(X)) > self.noise);
            %end
            
            [d dbp dnum] = Utils.checkgrad2_nodisplay(@(paramvec) cvae.fobj(paramvec, X), x0, 1e-5);            
            fprintf('diff=%g, norm(dbp-dnum)/norm(dbp)=%g\n', d, norm(Utils.vec(dbp-dnum))/(1e-8+norm(Utils.vec(dbp))));            
        end                        
        
        function [hprobs] = fprop(self,X)               
           self.dbn.setPar(size(X));
           self.dbn.fprop(X);
           hprobs = self.dbn.nnet{end}.OUT;
        end
        
        %function Xrec = reconstruct(self, hprobs)
        %    numdata = size(hprobs,2);
        %    Xrec = self.weights*hprobs + self.vbias*ones(1, numdata);
        %end
        
        %TODO: plot second layer, if no LCN, then simply linear combination 
        function [] = save(self) 
            %should add more items to learner_id in the future
            learner_id = sprintf('ConvAutoencoder_nu%d_l2%g_sp%g_spl%g',self.cnn.numunits, self.cnn.l2_C, self.target_sparsity, self.lambda_sparsity);
            savedir = fullfile(Config.basis_dir_path,self.save_dir);                       
            if ~exist(savedir,'dir')               
                mkdir(savedir);
            end
            savepath = fullfile(savedir, learner_id);
            
            learner = self;
            save([savepath '.mat'],'learner');
                        
            figure(1);
            Visualizer.displayByComposition2D(self.cnn.weights,self.lower_learner);
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
			            
            numunits = 100;
            type = 'gau';
            clear ae;
            ae = Autoencoder(numunits,type);	
            ae.save_iter = 1:ae.max_iter;
            ae.save_dir = 'test';
            ae.max_iter_per_batch = 3;
            ae.batch_size = 20000;
            ae.init_weight = 0.1;
            ae.l2_C = 0.01;
            % ae.noise = 0.5;
            ae.target_sparsity = 0.2;
            ae.lambda_sparsity = 5;
            
            % ae.checkGradient();
			ae.train(data.Xtrain(:,1:20000));
            
            clf;
            subplot(2,1,1);
            Utils.display_network_l1(data.Xtrain(:,1:6));
            subplot(2,1,2);
            Utils.display_network_l1(ae.reconstruct(ae.fprop(data.Xtrain(:,1:6))));
            saveas(gcf,'tmp.png');
		end
	end	   
       
end