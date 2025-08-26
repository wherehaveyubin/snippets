from PyPDF2 import PdfMerger

workspace = r'C:/directory/'

# Define file paths
ga_contract = workspace + "GA Contract.pdf"
bank_statement = workspace + "bank statement.pdf"
apartment = workspace + "apartment contract.pdf"
output_file = workspace + "merged_documents.pdf"

# Merge files in the requested order
merger = PdfMerger()
merger.append(ga_contract)
merger.append(bank_statement)
merger.append(apartment)
merger.write(output_file)
merger.close()

output_file
