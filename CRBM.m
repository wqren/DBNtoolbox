%WARNING!!: this code is obselete, currently only use static function

classdef CRBM < handle & RBM
    properties                
        spacing = 2;
    end
    methods
        function self = CRBM(numunits, type)      
            self = self@RBM(numunits, type);
            addpath(fullfile(Config.lib_dir_path,'conv2_ipp'));
            % addpath /home/honglak/umich/Library/convolution/IPP-conv2-mex/
			% self = self@RBM_v2(weights, hbias, vbias, type,l2_C, lambda_sparsity, target_sparsity);		            
        end
        
        %how to unify these methods?
        function train(self, X, par)                      
            %here batch_size means batch_ws, if need to change someday,
            %recon_error_sigma_epoch, and dW need to put batch_size into
            %consideration
            
            addpath(genpath('../utils'));  
            X = Data.Xtrain;
            self.numchannels = Data.numchannels;
            self.prev_learner = Data.learner;
            clear Data;                        
            
            ws = sqrt(self.feadim/self.numchannels);
           
            self.learner_id = savedir_s;
            
            if exist('sigma', 'var') && ~isempty(sigma) && sigma > 0   
                self.sigma = sigma;
                assign_sigma = true;
            else
				assign_sigma = false;
			end
            
            
            if self.type == 'bin'
                self.sigma = 1;
                assign_sigma = true;
            end
            
            %debug
            if exist('debug', 'var') && debug==true
                X = X(1:2);
            end
            
            if ~assign_sigma
                %estimate sigma by kmeans            
                patches = LoadData.get_random_patches_cell(X, ws,100000); 
                opt_approx = false; %true;
                if opt_approx
                    [label,center] = Kmeans.litekmeans(patches, min(self.numunits,500), true, 100);
                else
                    [label,center] = Kmeans.litekmeans(patches, self.numunits, true, 200);
                end

                sigma_kmeans = sqrt(mean(mean((patches- center(:, label)).^2,2)));
                err_kmeans = sum(mean( (patches- center(:, label)).^2, 2));
                err_kmeans

                self.sigma = sigma_kmeans;
            end
            %fixed parameters (currently using constant epsilon)
            eta_sigma = 0.01;
            initialmomentum  = 0.5;
            finalmomentum    = 0.9;
                      
            savedir = sprintf('/mnt/neocortex/scratch/suii/basis/%s',savedir_s);            
            
            if exist(savedir,'dir')
                disp('warning : directory already exist');
            else
                mkdir(savedir);
            end
        
           
            if ~self.optGPU
                Winc = zeros(ws*ws,self.numchannels, self.numunits);
                hbiasinc = zeros(size(self.hbias));
                vbiasinc = zeros(size(self.vbias));
            else
                error('GPU is not implemented yet')
                Winc = gzeros(ws*ws,self.numchannels, self.numunits);
                hbiasinc = gzeros(size(self.hbias));
                vbiasinc = gzeros(size(self.vbias));  
				self.weights = gsingle(self.weights);
				self.hbias = gsingle(self.hbias);
				self.vbias = gsingle(self.vbias); 		
            end
                    
            error_history = [];
            sparsity_history = [];
            error_for_sigma_history = [];
            
            % train a restricted Boltzmann machine
            runningavg_prob = [];
            timestamp = 10;
            
            max_iter = iter(end);
            iter = [0, iter];
            curr_iter_idx = 1;
            for t= 1 : max_iter                
                tic
                if t  > iter(curr_iter_idx)
                    if t ~= 1
                        DeepBeliefNetwork.save_progress(self, fname_mat, fname_png_1,fname_png_2, {error_history, sparsity_history}, {'error_history', 'sparsity_history'});
                    end

                    curr_iter_idx = curr_iter_idx + 1;
                    learner_id = sprintf('crbm_ws%d_h%d_%s_l2reg%g_eps%g_p%g_plambda%g_bs%g_sigma%g_iter%g', ws, self.numunits, self.type, self.l2_C, epsilon, self.target_sparsity, self.lambda_sparsity, batch_size,self.sigma, iter(curr_iter_idx));
                    fname_save = sprintf('%s/%s', savedir, learner_id);

                    fname_mat = sprintf('%s.mat', fname_save);
                    fname_png_1 = sprintf('%s_1.png', fname_save);
                    fname_png_2 = sprintf('%s_2.png', fname_save);
                    self.learner_id = [savedir_s '_' learner_id];
                end
                
                randidx = randperm(length(X));
                recon_err_epoch = [];
                sparsity_epoch = [];
                recon_err_for_sigma_epoch = 0;
                
				self.weights = reshape(self.weights, [ws*ws self.numchannels, self.numunits]);
                for b=1:length(X)                    
                    imdata = X{randidx(b)};
                    rows = size(imdata,2);
                    cols = size(imdata,1);
                    rowidx = ceil(rand*(rows-2*ws-batch_size))+ws + [1:batch_size];
                    colidx = ceil(rand*(cols-2*ws-batch_size))+ws + [1:batch_size];
                    
                    Xb = imdata(colidx, rowidx, :);
                    if self.optGPU
                        Xb = gsingle(Xb);
                    end
                    Xb = self.trim_image_for_spacing(Xb, ws);
                                                          
                    [~, poshidexp]  = self.inference(Xb);
                    [poshidstates poshidprobs] = self.mult_sample(poshidexp);
                    
                   posprods = self.gradient(Xb, poshidprobs, ws);
                   poshidact = squeeze(sum(sum(poshidprobs,1),2));

                    switch self.type
                        case 'gau'                          							
                            negdata = self.reconstruct(poshidstates);
                        case 'bin'            
%                             negdata = sigmoid((self.weights*poshidstates) + repmat(self.vbias, 1, size(Xb,2)));                            
                            error('binary data is not implemented yet')
                    end
                    
                    [~, neghidexp] = self.inference(negdata);
                    [~, neghidprobs] = self.mult_sample(neghidexp);
                    
                    negprods = self.gradient(negdata, neghidprobs, ws);
                    neghidact = squeeze(sum(sum(neghidprobs,1),2));

                    % monitoring variables
                    recon_err = double( mean( (Xb(:)-negdata(:)).^2 ));
                    recon_err_epoch = [recon_err_epoch recon_err];
                   
                    sparsity_epoch = [sparsity_epoch double(mean(poshidprobs(:)))];

                    % TODO: compute contrastive gradients
                    numcases = size(poshidprobs,1)*size(poshidprobs,2);
                    dW = posprods - negprods;
                    dhbias = poshidact - neghidact; 
                    dvbias = 0;
                   
                    
                    % TODO: sparsity regularization update
                    if self.lambda_sparsity > 0
                        % sparsity regularization update
                        if isempty(runningavg_prob)
                            runningavg_prob = squeeze(mean(mean(poshidprobs,1),2));
                        else
                            runningavg_prob = 0.9*runningavg_prob + 0.1*squeeze(mean(mean(poshidprobs,1),2));
                        end                        
                        dhbias_sparsity = self.lambda_sparsity*(repmat(self.target_sparsity,[length(runningavg_prob),1]) - runningavg_prob);
                    else
                        dhbias_sparsity = 0;
                    end
                    
                    dW = dW/numcases - self.l2_C*self.weights;
                    dhbias = dhbias/numcases + dhbias_sparsity;
                    
                    
                    if t<5,           
                        momentum = initialmomentum;
                    else
                        momentum = finalmomentum;
                    end

                    % update parameters                    
                    Winc = momentum*Winc + epsilon*dW;
                    self.weights = self.weights + Winc;

                    vbiasinc = momentum*vbiasinc + epsilon*dvbias;
                    self.vbias = self.vbias + vbiasinc;

                    hbiasinc = momentum*hbiasinc + epsilon*dhbias;
                    self.hbias = self.hbias + hbiasinc;

%                     fprintf('||W||=%g, ', double(sqrt(sum(self.weights(:).^2))));
%                     fprintf('epoch %d:\t error=%g,\t sparsity=%g\n', t, mean(recon_err_epoch), mean(sparsity_epoch));                    
                end
                self.weights = reshape(self.weights, [self.feadim, self.numunits]);
                
                error_history = [error_history mean(recon_err_epoch)];
                sparsity_history = [sparsity_history mean(sparsity_epoch)];

                if ~assign_sigma
                    sigma_recon = sqrt(error_history(end));
                    error_for_sigma_history = [error_for_sigma_history sigma_recon];
                end
                
                if ~assign_sigma
                    self.sigma = (1-eta_sigma)*self.sigma + eta_sigma*sigma_recon;                               
                end
                
                if mod(t,timestamp) == 1
					DeepBeliefNetwork.save_progress(self, fname_mat, fname_png_1,fname_png_2, {error_history, sparsity_history}, {'error_history', 'sparsity_history'});
                end
                elapse_time = toc;
                
                fprintf('||W||=%g, ', double(sqrt(sum(self.weights(:).^2))));
                fprintf('epoch %d:\t error=%g,\t sparsity=%g\n', t, error_history(t), sparsity_history(t));
                fprintf('elapsed time : %g\n', elapse_time);

            end
            DeepBeliefNetwork.save_progress(self, fname_mat, fname_png_1,fname_png_2, {error_history, sparsity_history}, {'error_history', 'sparsity_history'});
            
			if self.optGPU
				self.weights = double(self.weights);
				self.hbias = double(self.hbias);
				self.vbias = double(self.vbias);
            end            
        end

        function im2 = trim_image_for_spacing(self, im2, ws)
            % % Trim image so that it matches the spacing.
            spacing = self.spacing;
            if mod(size(im2,1)-ws+1, spacing)~=0
                n = mod(size(im2,1)-ws+1, spacing);
                im2(1:floor(n/2), : ,:) = [];
                im2(end-ceil(n/2)+1:end, : ,:) = [];
            end
            if mod(size(im2,2)-ws+1, spacing)~=0
                n = mod(size(im2,2)-ws+1, spacing);
                im2(:, 1:floor(n/2), :) = [];
                im2(:, end-ceil(n/2)+1:end, :) = [];
            end
        end
        
        function vishidprod2 = gradient(self,imdata, H, ws)
            numchannels = size(imdata,3);
            numbases = size(H,3);

            selidx1 = size(H,1):-1:1;
            selidx2 = size(H,2):-1:1;
            vishidprod2 = zeros(ws,ws,numchannels,numbases);

            if numchannels==1
                vishidprod2 = CRBM.conv2_ipp(imdata, H(selidx1, selidx2, :), 'valid');
            else
                for b=1:numbases
                    vishidprod2(:,:,:,b) = CRBM.conv2_ipp_boxcar(imdata, H(selidx1, selidx2, b), 'valid');
                end
            end
           
            vishidprod2 = reshape(vishidprod2, [ws^2, numchannels, numbases]);
        end  
        
        function [H HP Hc HPc] = mult_sample(self,poshidexp)
            % poshidexp is 3d array
            spacing = self.spacing;
            poshidprobs = poshidexp; %exp(poshidexp);
            poshidprobs_mult = zeros(spacing^2+1, size(poshidprobs,1)*size(poshidprobs,2)*size(poshidprobs,3)/spacing^2);
            poshidprobs_mult(end,:) = 0;
            % TODO: replace this with more realistic activation, bases..
            for c=1:spacing
                for r=1:spacing
                    temp = poshidprobs(r:spacing:end, c:spacing:end, :);
                    poshidprobs_mult((c-1)*spacing+r,:) = temp(:);
                end
            end

            % substract from max exponent to make values numerically more stable
            poshidprobs_mult = bsxfun(@minus, poshidprobs_mult, max(poshidprobs_mult,[],1));
            poshidprobs_mult = exp(poshidprobs_mult);

            [S1 P1] = CRBM.multrand2(poshidprobs_mult');
            S = S1';
            P = P1';
            clear S1 P1

            % convert back to original sized matrix
            H = zeros(size(poshidexp));
            HP = zeros(size(poshidexp));
            for c=1:spacing
                for r=1:spacing
                    H(r:spacing:end, c:spacing:end, :) = reshape(S((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
                    HP(r:spacing:end, c:spacing:end, :) = reshape(P((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
                end
            end

            if nargout >2
                Sc = sum(S(1:end-1,:));
                Pc = sum(P(1:end-1,:));
                Hc = reshape(Sc, [size(poshidexp,1)/spacing,size(poshidexp,2)/spacing,size(poshidexp,3)]);
                HPc = reshape(Pc, [size(poshidexp,1)/spacing,size(poshidexp,2)/spacing,size(poshidexp,3)]);
            end
        end
        
        function [poshidprobs2 poshidexp2] = inference(self, imdata)            
            W = self.weights;
            ws = sqrt(size(W,1));
            numbases = size(W,3);
            numchannel = size(W,2);

            if numchannel == 1
                H = reshape(W(end:-1:1, :, :),[ws,ws,numbases]);
                poshidexp2 = CRBM.conv2_ipp(imdata, H, 'valid');
                poshidprobs2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);
                for b=1:numbases
                    poshidexp2(:,:,b) = 1/(self.sigma).*(poshidexp2(:,:,b) + self.hbias(b));
                    poshidprobs2(:,:,b) = 1./(1 + exp(-poshidexp2(:,:,b)));
                end
            else % TODO: Multichannel cases were not tested yet
                if numchannel < numbases
                    poshidprobs2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);
                    poshidexp2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);
                    for c=1:numchannel
                        H = reshape(W(end:-1:1, c, :),[ws,ws,numbases]);
                        poshidexp2 = poshidexp2 + CRBM.conv2_ipp(imdata(:,:,c), H, 'valid');
                    end

                    for b=1:numbases
                        poshidexp2(:,:,b) = 1/(self.sigma).*(poshidexp2(:,:,b) + self.hbias(b));
                        poshidprobs2(:,:,b) = 1./(1 + exp(-poshidexp2(:,:,b)));
                    end
                else
                    poshidprobs2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);
                    poshidexp2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);
                    for b=1:numbases
                        H = reshape(W(end:-1:1, :, b),[ws,ws,numchannel]);
                        tempexp = CRBM.conv2_ipp_pairwise(imdata, H, 'valid');
                        poshidexp2(:,:,b) = 1/(self.sigma).*(sum(tempexp,3) + selfhbias(b));
                        poshidprobs2(:,:,b) = 1./(1 + exp(-poshidexp2(:,:,b)));
                    end
                end
            end

        end
        function negdata = reconstruct(self, S)
            W = self.weights;
            ws = sqrt(size(W,1));
            patch_M = size(S,1);
            patch_N = size(S,2);
            numchannels = size(W,2);
            numbases = size(W,3);

            % Note: Reconstruction was off by a few pixels in the original code (above
            % versions).. I fixed this as below:
            S2 = S;
            negdata2 = zeros(patch_M+ws-1, patch_N+ws-1, numchannels);
            if numchannels == 1
                H = reshape(W,[ws,ws,numbases]);
                negdata2 = sum(CRBM.conv2_ipp_pairwise(S2, H, 'full'),3);
            else
                for b = 1:numbases,
                    H = reshape(W(:,:,b),[ws,ws,numchannels]);
                    negdata2 = negdata2 + CRBM.conv2_ipp(S2(:,:,b), H, 'full');
                end
            end

            negdata =negdata2/self.sigma;
        end
    end
    methods(Static)
        function y = conv2_ipp(a, B, convopt)
            if strcmp(convopt, 'valid')
                y = double(ipp_mt_conv3(single(B),single(a), 'valid'));
            else
            y = double(ipp_mt_conv3(single(B),single(a), 'full'));
            if strcmp(convopt, 'same')
                z1x=size(a,1);
                z1y=size(a,2);
                z2x=size(B,1);
                z2y=size(B,2);
                px=((z2x-1)+mod((z2x-1),2))/2;
                py=((z2y-1)+mod((z2y-1),2))/2;

                y=double(y(px+1:px+z1x,py+1:py+z1y,:));
            end
            end        
        end
       
        function [S P] = multrand2(P)
            % P is 2-d matrix: 2nd dimension is # of choices

            % sumP = row_sum(P); 
            sumP = sum(P,2);
            P = P./repmat(sumP, [1,size(P,2)]);

            cumP = cumsum(P,2);
            % rand(size(P));
            unifrnd = rand(size(P,1),1);
            temp = cumP > repmat(unifrnd,[1,size(P,2)]);
            Sindx = diff(temp,1,2);
            S = zeros(size(P));
            S(:,1) = 1-sum(Sindx,2);
            S(:,2:end) = Sindx;

        end

        function y = conv2_ipp_boxcar(A, b, convopt)
            if strcmp(convopt, 'valid')
                y = double(ipp_mt_conv3(single(A),single(b), 'valid'));
            else
                y = double(ipp_mt_conv3(single(A),single(b), 'full'));
                if strcmp(convopt, 'same')
                    z1x=size(A,1);
                    z1y=size(A,2);
                    z2x=size(b,1);
                    z2y=size(b,2);
                    px=((z2x-1)+mod((z2x-1),2))/2;
                    py=((z2y-1)+mod((z2y-1),2))/2;

                    y=double(y(px+1:px+z1x,py+1:py+z1y,:));
                end
            end
        end
        function y = conv2_ipp_pairwise(A, B, convopt)
            if strcmp(convopt, 'valid')
                if size(A,1)>=size(B,1)
                    y = double(ipp_mt_conv2_pairwise(single(A),single(B), 'valid'));
                else
                    y = double(ipp_mt_conv2_pairwise(single(B),single(A), 'valid'));
                end
            else
                y = ipp_mt_conv2_pairwise(single(B),single(A), 'full');
                if strcmp(convopt, 'same')
                    z1x=size(A,1);
                    z1y=size(A,2);
                    z2x=size(B,1);
                    z2y=size(B,2);
                    px=((z2x-1)+mod((z2x-1),2))/2;
                    py=((z2y-1)+mod((z2y-1),2))/2;

                    y=double(y(px+1:px+z1x,py+1:py+z1y,:));
                end
            end
        end
        
    end
end