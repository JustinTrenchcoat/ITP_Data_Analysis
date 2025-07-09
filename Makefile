.PHONY: all clean plot

all: 
	make clean

plot:
	python omniPlot.py -s=$(s) -p=$(p) -tvd=$(tvd) -svd=$(svd)

# clean
clean: 
		rm -rf plots/*
		rm -rf gridDataMat/*
		rm -rf gridData/*
# rm -rf goldData/*