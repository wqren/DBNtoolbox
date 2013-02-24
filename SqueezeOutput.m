classdef SqueezeOutput < handle & NetworkLayer
%simply squeeze the output into feadim*numdata format 
%a trade-off between code cleaniness and efficiency (will adding a layer
%and copying IN/OUT affect a lot?)
	properties       
	end
	
	methods
		%those layers with parameters
		function [] = setPar(self,in_size)		
            self.in_size = in_size;
            self.out_size = prod(in_size);
        end		
		
        function object = gradCheckObject(self)                        
            object = self;
        end  
        
		%all layers must implement these
		function [] = fprop(self)            
			self.OUT = reshape(self.IN,[self.out_size, self.numdata]);
		end
        
		function [f derivative] = bprop(self,f,derivative)
			derivative = reshape(derivative,[self.in_size, self.numdata]);
		end
	end
end