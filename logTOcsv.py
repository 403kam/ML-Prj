import csv

#change these when needed. a-c is non-malware, 1-20 is malicious
input_file_path = './/log_files//2conn.log.labeled'
output_file_path = './/csv_files//2.csv'

with open(input_file_path, 'r') as input_file, open(output_file_path, 'w', newline='') as output_file:
  csv_writer = csv.writer(output_file, delimiter=',')

  #Get rid of header, and the last row. Then change value 20 to do T or F based on Beign or Malicous
  for _ in range(8):
    next(input_file)
  lines = list(input_file)[:-1]
  for line in lines:
    values = line.strip().split('\t')
    if values[20] not in ['-   Benign   -', '-   benign   -','(empty)   Benign   -']:
      values[20] = True
    else:
      values[20] = False
    
    csv_writer.writerow(values)

print('Conversion complete')