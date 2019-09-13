INSTALLATION := $(lastword $(sort $(wildcard /Applications/MATLAB_R*.app/bin/matlab)))

mex:
	@echo "Compiling using" $(INSTALLATION) "..."
	@$(INSTALLATION) -nodisplay -r " \
		cd('third_party/visionopencv/TemplateMatching'); \
		fprintf('\nAttempting to compile CPU version:\n'); \
		try mexOpenCV matchTemplateOCV.cpp; catch disp 'Compilation failed.'; end; \
		cd('../TemplateMatchingGPU'); \
		fprintf('\nAttempting to compile GPU PC version:\n'); \
		try mexOpenCV matchTemplateOCV_GPU.cpp -lgpu -lmwocvgpumex -largeArrayDims; catch disp 'Compilation failed.'; end; \
		fprintf('\nAttempting to compile GPU Linux/Mac version:\n'); \
		try mexOpenCV matchTemplateOCV_GPU.cpp -lmwgpu -lmwocvgpumex -largeArrayDims; catch disp 'Compilation failed.'; end; \
		exit"

clean:
	rm demo/*
	git checkout demo

	rm -f .*.mat
	rm -f .*.avi
