   classdef backgroundSubtractor
   %backgroundSubtractor Wrapper class for OpenCV class BackgroundSubtractorMOG2
   %   obj = backgroundSubtractor(history, varThreshold, bShadowDetection)
   %   creates an object with properties 
   %
   %   Properties:
   %   history          - Length of the history.
   %   varThreshold     - Threshold on the squared Mahalanobis distance
   %   bShadowDetection - Flag to enable/disable shadow detection
   %
   %   fgMask = getForegroundMask(obj, img) computes foreground mask on
   %   input image, img, for the object defined by obj.
   %
   %   reset(obj) resets object.
   %
   %   release(obj) releases object memory.

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %  Properties
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
       properties
           history = 500;
           varThreshold = single(4.0*4.0);
           bShadowDetection = true;
          
       end
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %  Public methods
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
       methods
           % Constructor
           function obj = backgroundSubtractor(history, varThreshold, bShadowDetection)
               if(nargin > 0)
                 obj.history          = history; 
                 obj.varThreshold     = varThreshold;
                 obj.bShadowDetection = bShadowDetection;                   
               else
                 obj.history          = 500; 
                 obj.varThreshold     = single(4^2);
                 obj.bShadowDetection = true;  
               end
               params = struct('history', obj.history, ...
                               'varThreshold', obj.varThreshold, ...
                               'bShadowDetection', obj.bShadowDetection);
               backgroundSubtractorOCV('construct', params);
           end

           % Get foreground mask
           function fgMask = getForegroundMask(~, img)

               % Get foreground mask
               fgMaskU8 = backgroundSubtractorOCV('compute', img);
               fgMask = (fgMaskU8 ~= 0);
           end
          
           % Reset object states
           function reset(obj)

               % Reset the background model with default parameters
               % This is done in two steps. First free the persistent
               % memory and then reconstruct the model with original
               % parameters
               backgroundSubtractorOCV('destroy');
               params = struct('history', obj.history, ...
                               'varThreshold', obj.varThreshold, ...
                               'bShadowDetection', obj.bShadowDetection);
               backgroundSubtractorOCV('construct', params);               
           end
           
           % Release object memory
           function release(~)
               % free persistent memory for model
               backgroundSubtractorOCV('destroy');
           end

       end
   end