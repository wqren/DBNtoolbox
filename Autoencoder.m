classdef Autoencoder < handle & Learner
%Autoencoder that uses minFunc to do optimization

%based on ICML2011 "On Optimization Methods for Deep Learning", LBFGS is better for low-dimensions input (compared with CG)
%and suggested number of iterations = 20, minibatch = 1000
                                  
    properties        
        weights;       %feadim*numunits
        hbias;         %numunits*1
        vbias;           %feadim*1
        
        type; %1:sigmoid-linear + sqaured error(gau), 2: sigmoid-sigmoid + cross-entropy (bin) 
        
        l2_C = 1e-4;        
        contractive_C = 0;;        
        target_sparsity = 0;
        lambda_sparsity = 0;
        noise = 0;                        
        init_weight = 1e-3;
        
        batch_size = 1000;        
        max_iter_per_batch = 20;
        
        numunits;
        feadim;                                                
                          
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
        function self = Autoencoder(numunits, type)                                               
            addpath(genpath(fullfile(Config.lib_dir_path,'minFunc')));            
            
            self.type = type;                        
            self.numunits = numunits;                                               
        end
                        
        function train(self, X)     
            %initialization
            self.feadim = size(X,1);
            if isempty(self.weights)
                self.weights = self.init_weight*randn(self.feadim,self.numunits);
                self.hbias = zeros(self.numunits,1);
                self.vbias = zeros(self.feadim,1);
                
                self.numParam = [numel(self.weights), numel(self.hbias), numel(self.vbias)];                
                self.numParam = cumsum(self.numParam);
                tstart = 0;
            else
                disp('use existing weights');
                tstart = length(self.error_history);
                self.save_iter = self.save_iter + tstart;
            end
                        
            options.Method = 'lbfgs';
            options.maxIter = self.max_iter_per_batch;       %max iteration for one batch
            
            numsamples = size(X,2);                
            
            self.l2_C_tmp = self.l2_C*self.batch_size;                                                
            
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
                    theta = self.getParam();
                    
                    %considering running avg. afterwards.
%                     if self.lambda_sparsity ~= 0 
%                         self.target_sparsity_tmp  =  self.target_sparsity;
%                     end
                                            
                    Xbatch = X(:,batch_idx);
                    if self.noise ~= 0 
                        self.mask = (rand(size(Xbatch)) > self.noise);                         
                    end

                    theta = minFunc(@(paramvec) self.fobj(paramvec, Xbatch), theta, options);                    
                    self.setParam(theta);
                    
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
            self.setParam(paramvec);                            
            
            if  self.noise ~= 0                        
               Xnoise = self.mask .* X; 
               hprobs = Utils.sigmoid(self.weights'*Xnoise + repmat(self.hbias, [1, self.batch_size]));                             
            else                
                hprobs = Utils.sigmoid(self.weights'*X + repmat(self.hbias, [1, self.batch_size]));                             
            end
                        
            ha = hprobs .* (1-hprobs);            
            Xrec = self.weights*hprobs + repmat(self.vbias, [1, self.batch_size]);
            
            if self.type == 'bin' 
                Xrec = Utils.sigmoid(Xrec); 
            end
            
            delta = Xrec - X;

            if self.type == 'gau'
                dyhat = 2*delta;
                f = sum(delta(:).^2);
            elseif self.type == 'bin'
                dyhat = delta;
                f = -sum(Utils.vec(X.*log(Xrec+1e-8) + (1-X).*log(1-Xrec+1e-8)));
            end
            self.recon_err = double(f/self.batch_size);
            self.sparsity = double(mean(hprobs(:)));
                        
            dWdec = dyhat * hprobs';
            
            dvbias = sum(dyhat,2);
            dhprobs = self.weights'*dyhat;
            
            if self.lambda_sparsity ~= 0 && self.target_sparsity > 0 
                %it's actually selectivity
                curr_sparsity = mean(hprobs, 2);
                tmp1 = self.target_sparsity./curr_sparsity;
                tmp2 = (1-self.target_sparsity)./(1-curr_sparsity);
                
                f = f + self.lambda_sparsity*self.batch_size*(sum(self.target_sparsity*log(tmp1) + (1-self.target_sparsity)*log(tmp2)));
                sparsityDelta = self.lambda_sparsity*(-tmp1+tmp2);
                dhprobs = dhprobs + repmat(sparsityDelta,1,self.batch_size);
            end            
                        
            hau = dhprobs .* ha;
            if self.noise ~= 0
                dWenc = hau * Xnoise';
            else
                dWenc = hau * X';               
            end
            dW = dWdec + dWenc'; % using tied weights

            dhbias = sum(hau ,2);
                                    
            
            if self.l2_C ~= 0
                f = f + 0.5*self.l2_C_tmp*sum(self.weights(:).^2);
                dW = dW + self.l2_C_tmp*self.weights;
            end

            if self.contractive_C ~= 0
                h2 = ha.^2; %sum over data
                one_minus_twoh = (1-2*hprobs);
                w2 = sum(self.weights.^2,1);
                J = sum(sum(h2,2).*w2');
                self.J = J/self.batch_size;
                f = f + 0.5 * self.contractive_C * J;  %already consider batch_size     
                
                dhbias = dhbias + self.contractive_C * w2' .* sum(h2 .* one_minus_twoh,2);                
                dW = dW + self.contractive_C * ( bsxfun(@times,X * (h2.*one_minus_twoh)' , w2) + bsxfun(@times,self.weights, sum(h2,2)')) ;
            end                       
            
            f = double(f);
            self.Obj = f;

            g = zeros(size(paramvec));
            
            g(1:self.numParam(1)) = double(dW(:));
            g(self.numParam(1)+1:self.numParam(2)) = double(dhbias(:));
            g(self.numParam(2)+1:end) = double(dvbias(:));                        
        end        
                
        function theta = getParam(self)
                theta = [self.weights(:); self.hbias(:); self.vbias(:)];
        end
        
        function setParam(self, paramvec)
                self.weights = reshape(paramvec(1:self.numParam(1)), size(self.weights));
                self.hbias = reshape(paramvec(self.numParam(1)+1:self.numParam(2)),size(self.hbias));
                self.vbias = paramvec(self.numParam(2)+1:end);
        end               
        
        function checkGradient(self)
            %for faster examination
            ori_feadim = self.feadim;
            ori_numunits = self.numunits;
            ori_batch_size = self.batch_size;
            ori_empty = isempty(self.weights);
            
            self.feadim = 3;
            self.numunits = 2;
            numsamples = 4;                        
            
            X = rand(self.feadim,numsamples);
            self.batch_size = numsamples;
            self.l2_C_tmp = self.l2_C*numsamples;
            self.weights = 1e-3*randn(self.feadim,self.numunits);
            self.hbias = zeros(self.numunits,1);
            self.vbias = zeros(self.feadim,1);               
            self.numParam = [numel(self.weights), numel(self.hbias), numel(self.vbias)];                            
            self.numParam = cumsum(self.numParam);
            x0 = self.getParam();
            
            if self.noise ~= 0
                self.mask = (rand(size(X)) > self.noise);
            end
            
            [d dbp dnum] = Utils.checkgrad2_nodisplay(@(paramvec) self.fobj(paramvec, X), x0, 1e-5);            
            fprintf('diff=%g, norm(dbp-dnum)/norm(dbp)=%g\n', d, norm(Utils.vec(dbp-dnum))/(1e-8+norm(Utils.vec(dbp))));
            
            self.feadim = ori_feadim;
            self.numunits = ori_numunits;
            self.batch_size = ori_batch_size;
            if ori_empty
                self.weights = [];
                self.hbias = [];
                self.vbias = [];
            end
        end                        
        
        function [hprobs] = fprop(self,X)            
            numdata = size(X,2);
            hprobs = Utils.sigmoid(self.weights'*X + self.hbias*ones(1,numdata));  %numunits * numsamples            
        end
        
        function Xrec = reconstruct(self, hprobs)
            numdata = size(hprobs,2);
            Xrec = self.weights*hprobs + self.vbias*ones(1, numdata);
        end
        
            
        function [] = save(self) 
            %should add more items to learner_id in the future
            learner_id = sprintf('Autoencoder_nu%d_l2%g_sp%g_spl%g',self.numunits, self.l2_C, self.target_sparsity, self.lambda_sparsity);
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