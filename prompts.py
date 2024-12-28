BIO_TRIPLETS_PROMPT = """You are a helpful assistant, you are given a text, please extract the triplets from the text. There will be two examples:

Example 1:
Text: Various methods of diagnosing allergic factors in chronic rhinitis are discussed. Among the procedures which aim at detecting specific allergens, i.e. skin testing, RAST, and nasal provocation tests, the last mentioned, as they are performed directly on the shock organ, have so far been found to give the most accurate picture of clinically dominant allergens and of the intensity and character of the rhinitis. However, information obtained by analysing the correlations between different procedures is not unanimous. As long as test techniques and allergen extracts have not been standardized, one particular test cannot be recommended as the method of choice.

Triplets:(skin testing, aims at detecting, specific allergens)  
(RAST, aims at detecting, specific allergens)  
(nasal provocation tests, aims at detecting, specific allergens)  
(nasal provocation tests, performed directly on, shock organ)  
(nasal provocation tests, give, accurate picture of clinically dominant allergens)  
(nasal provocation tests, give, accurate picture of intensity and character of rhinitis)  
(correlations, analysed between, different procedures)  
(test techniques, lack, standardization)  
(allergen extracts, lack, standardization)  
(test techniques, not recommended as, method of choice)  


Example 2:
Text: KRAS genes belong to the most frequently mutated family of oncogenes in cancer. The G12C mutation, found in a third of lung, half of colorectal and pancreatic cancer cases, is believed to be responsible for a substantial number of cancer deaths. For 30 years, KRAS has been the subject of extensive drug-targeting efforts aimed at targeting KRAS protein itself, but also its post-translational modifications, membrane localization, protein-protein interactions and downstream signalling pathways. So far, most KRAS targeting strategies have failed, and there are no KRAS-specific drugs available. However, clinical candidates targeting the KRAS G12C protein have recently been developed. MRTX849 and recently approved Sotorasib are covalent binders targeting the mutated cysteine 12, occupying Switch II pocket.Herein, we describe two fragment screening drug discovery campaigns that led to the identification of binding pockets on the KRAS G12C surface that have not previously been described. One screen focused on non-covalent binders to KRAS G12C, the other on covalent binders.

Triplets:
(KRAS genes, belong to, oncogenes)  
(KRAS genes, frequently mutated in, cancer)  
(G12C mutation, found in, lung cancer)  
(G12C mutation, found in, colorectal cancer)  
(G12C mutation, found in, pancreatic cancer)  
(G12C mutation, responsible for, cancer deaths)  
(KRAS, subject of, drug-targeting efforts)  
(drug-targeting efforts, aimed at targeting, KRAS protein)  
(drug-targeting efforts, targeted, post-translational modifications)  
(drug-targeting efforts, targeted, membrane localization)  
(drug-targeting efforts, targeted, protein-protein interactions)  
(drug-targeting efforts, targeted, downstream signalling pathways)  
(KRAS targeting strategies, failed, most cases)  
(KRAS-specific drugs, unavailable for, KRAS)  
(clinical candidates, developed for, KRAS G12C protein)  
(MRTX849, targets, mutated cysteine 12)  
(Sotorasib, targets, mutated cysteine 12)  
(Sotorasib, approved for, KRAS G12C protein)  
(fragment screening campaigns, identified, new binding pockets)  
(binding pockets, located on, KRAS G12C surface)  
(one screen, focused on, non-covalent binders)  
(other screen, focused on, covalent binders)  

Now, please extract the triplets from the following text:

Text: {text}

Triplets:
"""