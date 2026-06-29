version -1	not modularized process	
version 0	modularized process	
version 0.1	improvement	
	   	Added Cleaner function	
	   	Added Logger function	
	   	Added Testing File for Retriever	
	   	Improved Prompt	
	   	Added "time and date" in user 	
	      		and assistant response	
	   	Fixed the alignment of the 	
	      		assistant response	
	   	Remove the "top k" slider in UI	
	      		and its corresponding code	
	   	Remove chunk generation in UI	
	      		and its corresponding code	
	   	Added "Generating response" 	
	      		spinner after "Searching documents"	
	      		spinner	
	   	Added source filename in assistant	
	      		response	
	   	Added response loop if there is no 	
	      		results found. Asking user if query	
	      		is correct or has a typographical 	
	      		error.
version 0.2	improvement
		Added Memory chat
		Remove "load Indxed" button
				incorporate the indexed loading
				upon streamlit execution
		Remove the "Chunks Information" box
		Fixed the "source file" not visible in
				assistant response history
version 0.3	improvement
		Added "New Chat" functionality
		Added "Recent Chat" functionality
		Transfer "Index Documents" functionality
				inside the program. It also
				detects if there are new documents
				in the repository and create vector
				for it (incremental creation)
		Added "Keep Alive" in the LLM operation
