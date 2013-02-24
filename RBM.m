classdef RBM < handle & Learner
    %parent class for RBM-like model
	%doing update in another function (will this affect the speed a lot?)    
	
    properties                
        weights;       %feadim*numunits                
        hbias;
        vbias;
        
        numunits;
        feadim;        
        
        %parameters
        l2_C = 1e-4;		
        lambda_sparsity = 0;
        target_sparsity = 0;                		
        type;  %'bin' or 'gau'        		        
        
        batch_size= 100;		
        sigma = 1;		
        epsilon = 1e-3;
        initialmomentum = 0.5;
        finalmomentum = 0.9;
        init_weight = 1e-3;
        
        error_history;
        sparsity_history;        		
		
        %temporary varaibles
		Winc;
		hbiasinc;
		vbiasinc;
		runningavg_prob;		
    end
    
    methods
        function self = RBM(numunits, type)                                    
            self.numunits = numunits;            
            self.type = type;                     
        end
        
        function train(self, X, par)
            if ~exist('par','var') %addtional parameters for child class
                par = [];
            end
            
            if isempty(self.save_iter)
                self.save_iter = self.max_iter;
            end
            
            %initialization
            self.feadim = size(X,1);
            if isempty(self.weights)
                self.weights = self.init_weight*randn(self.feadim,self.numunits);
                self.hbias = zeros(self.numunits,1);
                self.vbias = zeros(self.feadim,1);
                
                self.Winc = zeros(size(self.weights));
                self.hbiasinc = zeros(size(self.hbias));
                self.vbiasinc = zeros(size(self.vbias));
                tstart = 0;
            else
                disp('use existing weights');
                tstart = length(self.error_history);
                self.save_iter = self.save_iter + tstart;
            end
                        		                                
            
            % train a restricted Boltzmann machine
            runningavg_prob = [];
                                    
            for t= tstart+1 : tstart+self.max_iter                                
                tic                                
                randidx = randperm(size(X,2));
                recon_err_epoch = [];
                sparsity_epoch = [];                                
				
                for b=1:floor(size(X,2)/self.batch_size)  %//一次只做一個batch
                    batchidx = randidx((b-1)*self.batch_size+1:min(b*self.batch_size, size(X,2))); %random choose batch

                    Xb = X(:, batchidx);                   
					                    					
                    % TODO: compute contrastive divergence steps                     
                    [recon_err sparsity] = self.update(Xb,t,par);
					
                    recon_err_epoch = [recon_err_epoch recon_err];                    
                    sparsity_epoch = [sparsity_epoch sparsity];                          
                end
                
                self.error_history = [self.error_history mean(recon_err_epoch)];
                self.sparsity_history = [self.sparsity_history mean(sparsity_epoch)];
               
                elapse_time = toc;
                
                fprintf('||W||=%g, ', double(sqrt(sum(self.weights(:).^2))));
                fprintf('epoch %d:\t error=%g,\t sparsity=%g\n', t, self.error_history(end), self.sparsity_history(end));
                fprintf('elapsed time : %g\n', elapse_time);
                
                %save if needed
                if nnz(self.save_iter == t) > 0
                    self.save();
                end
            end
        end
               
        function [recon_err sparsity] = update(self, Xb, epoch, par)			
			poshidprob = (self.weights'*Xb + self.hbias*ones(1,self.batch_size))/self.sigma ;  							  
			poshidprob = Utils.sigmoid(poshidprob);
			poshidstates = poshidprob > rand(self.numunits,self.batch_size);                   
			
			switch self.type
				case 'gau'                          
					negdata = (repmat(self.vbias,[1, self.batch_size])+self.weights*poshidstates)/self.sigma;
				case 'bin'            
					negdata = Utils.sigmoid((self.weights*poshidstates) + repmat(self.vbias, 1, size(Xb,2)));                            
				otherwise
					error('undefined type');
			end
			
			neghidprob = (self.weights'*negdata + self.hbias*ones(1,self.batch_size))/self.sigma ;                     			 
			neghidprob = Utils.sigmoid(neghidprob);
			
			% monitoring variables
			recon_err = double(norm(Xb- negdata, 'fro')^2/size(Xb,2));			
			
			sparsity = double(mean(mean(poshidprob)));
			
			% TODO: compute contrastive gradients
			dW = Xb*poshidprob'- negdata*neghidprob';
			dhbias = sum(poshidprob,2) - sum(neghidprob,2); 
			dvbias = sum(Xb,2)- sum(negdata,2);
		   			
			% TODO: sparsity regularization update
			if self.lambda_sparsity>0 && self.target_sparsity> 0
				% sparsity regularization update
				if isempty(self.runningavg_prob)
					self.runningavg_prob = mean(poshidprob,2);
				else
					self.runningavg_prob = 0.9*self.runningavg_prob + 0.1*mean(poshidprob,2);
				end
				dhbias_sparsity = self.lambda_sparsity*(repmat(self.target_sparsity,[length(self.runningavg_prob),1]) - self.runningavg_prob);
			else
				dhbias_sparsity = 0;
			end
			
			dW = dW/self.batch_size - self.l2_C*self.weights;
			dvbias = dvbias/self.batch_size;
			dhbias = dhbias/self.batch_size + dhbias_sparsity;
							                            
			% update parameters						            					            
            if epoch<5,           
                momentum = self.initialmomentum;
            else
                momentum = self.finalmomentum;
            end

			self.Winc = momentum*self.Winc + self.epsilon*dW;
			self.weights = self.weights + self.Winc;

			self.vbiasinc = momentum*self.vbiasinc + self.epsilon*dvbias;
			self.vbias = self.vbias + self.vbiasinc;

			self.hbiasinc = momentum*self.hbiasinc + self.epsilon*dhbias;
			self.hbias = self.hbias + self.hbiasinc;
		end
				
		
        function [acti] = fprop(self, X)            
                acti = Utils.sigmoid((self.weights'*X + repmat(self.hbias,[1,size(X,2)])) /self.sigma); 
        end                
        
        function [FE] = freeEnergy(self, X)
            %compute negative free energy -F(x)
            tmp = self.weights'*X + repmat(self.hbias, [1, size(X,2)]);
            tmp(tmp<20) = log(exp(tmp(tmp<20))+1);
            FE = (sum(tmp,1) + self.vbias'*X)';
        end
        
        function [] = save(self)
            %should add more items to learner_id in the future
            learner_id = sprintf('RBM_nu%d_l2%g_sp%g_spl%g',self.numunits, self.l2_C, self.target_sparsity, self.lambda_sparsity);
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
            
            %currently only show obj, error & sparsity
            clf;            
            subplot(2,1,1);
            plot(self.error_history);
            subplot(2,1,2);
            plot(self.sparsity_history);
            saveas(gcf,[savepath '_2.png']);
        end
    end
      
	methods(Static)
		function [] = test_natural_image()			
			disp('test with natural image')
			data = DataLoader();
			data.loadNaturalImage();
			
            %(these parameters are actually hard to tune)
            numunits = 100;
            type = 'gau';
            clear rbm;
            rbm = RBM(numunits,type);	
            rbm.save_iter = 1:rbm.max_iter;
            rbm.save_dir = 'test';
            rbm.l2_C = 0;
            rbm.epsilon = 0.05;
            rbm.init_weight = 0.1;
            
            % ae.checkGradient();
			rbm.train(data.Xtrain(:,1:20000));
		end
	end	
end