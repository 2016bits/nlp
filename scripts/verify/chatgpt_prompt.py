FEVER_PROMPT = """Verify the claim according to the evidence titles and sentences.Several examples are given as follows.

claim: Roman Atwood is a content creator.
evidence: <title> Roman Atwood <sentence> He is best known for his vlogs , where he posts updates about his life on a daily basis .
label: SUPPORTS

claim: Adrienne Bailon is an accountant.
evidence: <title> Adrienne Bailon <sentence> Adrienne Eliza Houghton ( Bailon ; born October 24 , 1983 ) is an American singer-songwriter , recording artist , actress , dancer and television personality .
label: REFUTES

claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
evidence: <title> Nikolaj Coster-Waldau <sentence> He then played Detective John Amsterdam in the short-lived Fox television series New Amsterdam ( 2008 ) , as well as appearing as Frank Pike in the 2009 Fox television film Virtuality , originally intended as a pilot . <title> Fox Broadcasting Company <sentence> The Fox Broadcasting Company ( often shortened to Fox and stylized as FOX ) is an American English language commercial broadcast television network that is owned by the Fox Entertainment Group subsidiary of 21st Century Fox .
label: SUPPORTS

claim: Peggy Sue Got Married is a Egyptian film released in 1986.
evidence: <title> Peggy Sue Got Married <sentence> Peggy Sue Got Married is a 1986 American comedy-drama film directed by Francis Ford Coppola starring Kathleen Turner as a woman on the verge of a divorce , who finds herself transported back to the days of her senior year in high school in 1960 . <title> Francis Ford Coppola <sentence> Francis Ford Coppola ( born April 7 , 1939 ) , also credited as Francis Coppola , is a semi-retired American film director , producer , and screenwriter .
label: REFUTES

claim: System of a Down briefly disbanded in limbo.
evidence: <title> System of a Down (album) <sentence> System of a Down is the debut studio album by Armenian American metal band System of a Down , released on June 30 , 1998 , by American Recordings and Columbia Records . <title> Limbo (video game) <sentence> Limbo is a puzzle platform video game developed by independent studio Playdead . <title> System of a Down <sentence> System of a Down , sometimes shortened to SOAD or System , is an heavy metal band from Glendale , California , formed in 1994 . <title> Limbo (video game) <sentence> The game was released in July 2010 on Xbox Live Arcade , and has since been ported to several other systems , including the PlayStation 3 and Microsoft Windows . <title> Limbo (video game) <sentence> Limbo received positive reviews , but its minimal story polarised critics ; some critics found the open ended work to have deeper meaning that tied well with the game 's mechanics , while others believed the lack of significant plot and abrupt ending detracted from the game .
label: NOT ENOUGH INFO

claim: 
evidence: <title>  <sentence> 
label: 

"""