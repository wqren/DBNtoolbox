classdef LCN < handle & NetworkLayer
%Local Contrast Normalization, currently design for 2D data (ex: image),
%and only support convolution (stride = 1)
	properties
       ws;
       kernel; %gaussian kernel
       
       ws_rad = 4; %std for gaussian kernel
       gamma = 0.01; %constant for sigma map to prevent numerical error
       
       %temporary variable       
       sqrt_sigma;       
       mask;
       border_bias; %the output shiftment due to valid convolution
	end
	
	methods
        function self = LCN(ws)
           self.ws = ws;             
           self.border_bias = ceil(ws/2);
        end
        
		function [] = setPar(self,in_size)		
            self.in_size = in_size;
            %do valid convolution, since hidden units in boundary are
            %biased from boundary effect anyway
            self.out_size = [in_size(1)-self.ws+1, in_size(2)-self.ws+1, in_size(3)]; 
                        
            %divide numchannels later for stability?
            self.kernel = fspecial('gaussian',self.ws,self.ws/self.ws_rad) / in_size(3);           
            if Config.gpu %this should be a special case
                self.kernel = gpuArray(self.kernel);
            end
        end		
		
        function object = gradCheckObject(self)                        
            ws = 2;
            object = LCN(ws);
        end  
        
		%all layers must implement these
		function [] = fprop(self)		
            sqr = self.IN.^2;
            mu = Utils.zeros([self.out_size,self.numdata]);
            sigma = Utils.zeros([self.out_size,self.numdata]);

            %convn is sligthly faster than conv2
            for d = 1 : self.numdata
                mu(:,:,:,d) = convn(self.IN(:,:,:,d),self.kernel,'valid');
                sigma(:,:,:,d) = convn(sqr(:,:,:,d),self.kernel,'valid');
            end            
            
            mu = sum(mu,3);
            sigma = sum(sigma,3);
            
            self.OUT = self.IN(self.border_bias:self.border_bias+self.out_size(1)-1,...
                self.border_bias:self.border_bias+self.out_size(2)-1,:,:) - repmat(mu,[1 1 self.in_size(3), 1]);            
            
            self.sqrt_sigma = sqrt(sigma);
                       
            self.mask = self.sqrt_sigma > self.gamma;  
            self.sqrt_sigma = max(self.sqrt_sigma,self.gamma);
                        
            self.OUT = self.OUT ./ repmat(self.sqrt_sigma,[1 1 self.in_size(3), 1]);     
		end
        
		function [f derivative] = bprop(self,f,derivative)
			if ~self.skip_passdown
                dX = Utils.zeros([self.in_size, self.numdata]);
                frac_dY_M = derivative./repmat(self.sqrt_sigma,[1 1 self.in_size(3), 1]);%M is denominator
                dX(self.border_bias:self.border_bias+self.out_size(1)-1,...
                self.border_bias:self.border_bias+self.out_size(2)-1,:,:) = frac_dY_M;
                
                sum_derivative = sum(frac_dY_M,3);
                for d = 1 : self.numdata
                    dX(:,:,:,d) = dX(:,:,:,d) - repmat(conv2(self.kernel,sum_derivative(:,:,1,d),'full'),[1 1 self.in_size(3)]);
                end
                                
                dX2 = Utils.zeros([self.in_size, self.numdata]); 
                sum_derivative2 = sum(frac_dY_M.*self.OUT,3);
                for d = 1 : self.numdata
                    tmp = self.mask(:,:,1,d) .* sum_derivative2(:,:,1,d)./self.sqrt_sigma(:,:,1,d);                    
                    dX2(:,:,:,d) = repmat(conv2(self.kernel,tmp,'full'),[1 1 self.in_size(3)]);                                        
                end
                dX2 = dX2 .* self.IN;
                
                derivative = dX - dX2;
            end   
		end
	end
end