#CreateModel
/Model/trained_pipeline_LogisticRegression.pkl: Data/dfAttacksX.csv
        python src/makemodel.py --endpoint_path=Data/dfAttacksX.csv --output_path=Models/

#Predict
data/resultsTable.csv: data/dfAttacksX.csv
	python src/predictY.py --endpoint_path=Data/dfAttacksX.csv --output_path=Data/

# Clean the data
clean:
	rm -rf Data/*
	rm -rf Models/*
	