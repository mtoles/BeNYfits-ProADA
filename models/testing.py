import nltk
nltk.download('punkt')

nltk.sent_tokenize({'Talking to cats': 'Quirky', 'Fastidious folding clothes': 'Quirky', 'Flustered easily in social situations': 'Quirky', "Sleeping with arm under SO's clothes": 'Quirky', "Gets very 'comfortable' quickly": 'Quirky', 'Direct about future visions': 'Quirky', "Bit of a 'wild' past": 'Context Dependent', 'Talking to herself': 'Quirky', 'Ordering/alphabetizing everything': 'Quirky', 'Special containers and ordering in fridge': 'Quirky', 'Brushes teeth 3 or 4 times a day': 'Quirky', 'Very forthright about personal hygiene': 'Quirky'})