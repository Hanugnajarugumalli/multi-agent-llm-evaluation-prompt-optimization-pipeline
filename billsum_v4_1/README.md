BillSum Dataset

**NOTE:** This dataset was updated on 12/3/2019 - please ignore any older versions of the data.

*Minor update on 2/26/2019* - to fix how CA file is formatted.

Each file is in a jsonlines format: each line is a json containing data for one bill.


Each json has the following fields: 
	- text: bill text
	- clean_text: a preprocessed copy of the text (with whitespace formatting removed and other basic cleaning. )
	- summary: bill summary
	- title: bill title
	- bill id: An identified for the bill - in US data it is SESSION_BILL-ID, for CA BILL-ID (since all bills are from the 2015-2016 session) 