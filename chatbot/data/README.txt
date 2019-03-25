Cornell Movie-Dialogs Corpus


A) Brief description:

This corpus contains a metadata-rich collection of fictional conversations extracted from raw movie scripts:

- 220,579 conversational exchanges between 10,292 pairs of movie characters
- involves 9,035 characters from 617 movies
- in total 304,713 utterances


B) Files description:

In all files the field separator is " +++$+++ "


- movie_lines.txt
	- contains the actual text of each utterance
	- fields:
		- lineID
		- characterID (who uttered this phrase)
		- movieID
		- character name
		- text of the utterance

- movie_conversations.txt
	- the structure of the conversations
	- fields
		- characterID of the first character involved in the conversation
		- characterID of the second character involved in the conversation
		- movieID of the movie in which the conversation occurred
		- list of the utterances that make the conversation, in chronological 
			order: ['lineID1','lineID2',É,'lineIDN']
			has to be matched with movie_lines.txt to reconstruct the actual content



Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)