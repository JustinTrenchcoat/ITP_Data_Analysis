.PHONY: all clean plot cleanplot

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

cleanplot:
	rm -rf plots/bulk/Histograms/*
	rm -rf plots/bulk/Boxplots/*
	rm -rf plots/fine/Histograms/*
	rm -rf plots/fine/Boxplots/*
