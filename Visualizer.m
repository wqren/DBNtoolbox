classdef Visualizer 
    methods(Static)
        function [Xstar] = displayByOptimization2D(dbn, unit_idx)    
            error('not finish implementation');
            %TODO: add constraint on input's scale!!
            %TODO: multiple unit_idx at a time?
            %for now, the last layer of dbn must be squeezed          
            
            dbn.setNumData(1);
            
            dbn.nnet{1}.skip_passdown = false;
            x0 = Utils.vec(randn(dbn.nnet{1}.in_size));
            Xstar = minFunc(@(paramvec) Visualizer.fobjVisualize(paramvec, dbn, unit_idx), x0);                                       
            
            Xstar = reshape(Xstar, dbn.nnet{1}.in_size);
            if dbn.nnet{1}.in_size(3) == 1
                colormap(gray);
            end
% 			numcol = round(sqrt(numunits2));
% 			numrow = numunits2 / numcol;
            numcol = 1;
            numrow = 1;
			for i = 1 : length(unit_idx)
				subplot(numcol, numrow, i);
                cmax=max(Utils.vec(Xstar));
                cmin=min(Utils.vec(Xstar));
				imagesc((Xstar-cmin)/(cmax-cmin)); %check if it's the correct function to use
            end     
            dbn.nnet{1}.skip_passdown = true;
        end
        
        function [f g] = fobjVisualize(paramvec,dbn,unit_idx)
            %objective function for visualization based on optimization            
            paramvec = reshape(paramvec, dbn.nnet{1}.in_size);
            dbn.fprop(paramvec);
            gamma = 1e4; % a constant to make gradient not too small
            f = -dbn.nnet{end}.OUT(unit_idx)*gamma;
            dx = zeros(dbn.nnet{end}.out_size,1);
            dx(unit_idx) = -gamma;
            [~, g] = dbn.bprop(dx);
            g = g(:);
        end
        
        function [weights] = displayByComposition2D(weights, lower_learner)
  %pool: the Pooling2D object used to pool from lower_weights into weights
  %cell: weights&pool from high to low
            if exist('lower_learner', 'var') && ~isempty(lower_learner)
                if ~iscell(lower_learner)
                    lower_learner = {lower_learner};
                end
                for i = 1 : length(lower_learner)            
                    [ws2 ws2 numunits numunits2] = size(weights);
                    [ws ws numchannels numunits] = size(lower_learner{i}.cnn.weights);

                    pool_curr = Pooling2D(lower_learner{i}.pool.ws,lower_learner{i}.pool.stride,'avg');
                    sim_in_size = (ws2-1)*lower_learner{i}.pool.stride+lower_learner{i}.pool.ws;
                    pool_curr.setPar([sim_in_size,sim_in_size,numunits]);
                    pool_curr.numdata = numunits2;

                    [~, lower_map] = pool_curr.bprop(0,weights);
                    weights = Utils.convReconstruct(lower_map, lower_learner{i}.cnn.weights);                
                end
            else
                [ws ws numchannels numunits2] = size(weights);
            end
            
            if numchannels == 1
                colormap(gray);
            end
			numcol = round(sqrt(numunits2));
			numrow = numunits2 / numcol;
			for i = 1 : numunits2
				subplot(numcol, numrow, i);
                cmax=max(Utils.vec(weights(:,:,:,i)));
                cmin=min(Utils.vec(weights(:,:,:,i)));
				imagesc((weights(:,:,:,i)-cmin)/(cmax-cmin)); %check if it's the correct function to use
            end                            
        end
        
         function display_network_l1(A,numcols,numchannels,figstart)
            %display_network -- assume filter is for 2D image

            warning off all

            if exist('figstart', 'var') && ~isempty(figstart), figure(figstart); end

            [L M]=size(A);  %feadim, numunits
            
            if ~exist('numchannels', 'var') || isempty(numchannels)
                numchannels = 1;
            end
            
            L = L / numchannels;
            A = reshape(A, [L, numchannels, M]);
            
            if ~exist('numcols', 'var') || isempty(numcols)
                numcols = ceil(sqrt(L));
                while mod(L, numcols), numcols= numcols+1; end
            end
            ysz = numcols;
            xsz = ceil(L/ysz);

            m=floor(sqrt(M*ysz/xsz));
            n=ceil(M/m);

            if numchannels == 1
                colormap(gray);
            end

            buf=1;
            if numchannels == 1
                array=-ones(buf+m*(xsz+buf),buf+n*(ysz+buf));
            else 
                array=zeros(buf+m*(xsz+buf),buf+n*(ysz+buf),numchannels);
            end
            k=1;
            for i=1:m
                for j=1:n
                    if k>M continue; end
                    if numchannels == 1
                        clim=max(abs(A(:,:,k)));
                        array(buf+(i-1)*(xsz+buf)+[1:xsz],buf+(j-1)*(ysz+buf)+[1:ysz])= reshape(A(:,:,k),[xsz,ysz])/clim;
                    else
                        cmax=max(max(A(:,:,k)));
                        cmin = min(min(A(:,:,k)));
                        array(buf+(i-1)*(xsz+buf)+[1:xsz],buf+(j-1)*(ysz+buf)+[1:ysz],:)= (reshape(A(:,:,k),[xsz,ysz, numchannels])-cmin) /(cmax-cmin) ;                        
                    end
                    k=k+1;
                end
            end

            if isreal(array)
                h=imagesc(array,'EraseMode','none',[-1 1]);
            else
                h=imagesc(20*log10(abs(array)),'EraseMode','none',[-1 1]);
            end;
            axis image off

            drawnow

            warning on all            
        end
    end
end