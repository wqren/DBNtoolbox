classdef NetworkLayer < handle
	properties
		in_size; 
        out_size; 
        numdata;
		
        skip_update = false;    
        skip_passdown = false;                
        		
        paramNum=0;
        
        %store temporary data
        IN;                
        OUT;            
	end
	
	methods
		%those layers with parameters
		function [] = setPar(self,in_size, out_size)		
            error('need implement this!')
		end
		
        function [] = reset(self)            
        end
		
		function clearTempData(self)           
            self.IN = [];
            self.OUT = [];
        end        		
		
        function param = getParam(self)	       
           param = [];
        end
   
        function param = getGradParam(self)	        
            param = [];
        end
        
        function setParam(self,paramvec)            
        end
		
        function object = gradCheckObject(self)                        
        end  
        
		%all layers must implement these
		function [] = fprop(self)
			error('need implement this!')
		end
        
		function [f derivative] = bprop(self,f,derivative)
            if ~self.skip_update %update the parameters
            end
            if ~self.skip_passdown %compute gradient for lower layers
            end
			error('need implement this!')
		end
	end
end