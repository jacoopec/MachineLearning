• Mescolo i dati del dataset
• Splitto randomicamente il dataset D in K blocchi B1, . . . , Bk
• ∀k ∈ {1, . . . , K} Te = Bk e Tr = D − Bk (rimanenti k − 1 blocchi)
• Imparo ogni modello Mk su Tr
• Testo ogni modello su Te valutandone le performance
• Il valore finale della performance sar`a la media delle singole performance