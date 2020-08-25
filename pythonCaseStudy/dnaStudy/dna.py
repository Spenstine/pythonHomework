def read_seq(inputfile):
    '''
    read textfile and return a string sequence 
    of the content
    '''
    with open(inputfile, "r") as fin:
        seq = fin.read()

    #remove unwanted characters from the dna sequence
    seq = seq.replace('\n', '')
    seq = seq.replace('\r', '')
    seq = seq.replace(' ', '')
    return seq





#extract dna sequence
dna = read_seq('dna.txt')

#extract protein sequence
prt = read_seq('protein.txt')


def translate(stringOfDna):
    '''
    Translate a string containing a nucleotide sequence into a string 
    containing the corresponding sequence of amino acids.
    Nucleotides are translated in triplets using the table dictionary;
    each amino acid 4 is encoded with a string of length 1.
    '''

    #dict table:
    #dna->amino acid
    table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }

    #protein is a string of amino acids
    protein = ""
 
    #if stringOfDna % 3 == 0:
    #    pass
 
    #one amino acid(codon) consists of three dna in sequence
    for i in range(0, len(stringOfDna), 3):
        codon = stringOfDna[i : i + 3].upper()
        if len(codon) == 3:
            protein += table[codon]

    return protein

'''
we translate from 21 to 938 based on NCBI.
however, python is zero indexed, so to get the 
21st character we use index 20, and the 938th with index 937.
because slicing excludes the stop point we stop at index 938 to
have 937 as the last index in the sequence.
'''
proteinTranslated = translate(dna[20:938])

#exclude the last character since it signals the end of the protein sequence.
print(proteinTranslated[:-1] == prt)