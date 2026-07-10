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
		Remove "load Indxed" button and
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
version 0.4	impprovement
		Added rewriter for memory chat
		Added "Clear Chat" functionality
		Improved the user interface
		Added branding and tag line
		Remove the welcome message upon
				submission of first query
		Disabled the new and clear chat buttons
				upon generating the assistant
				response
		Disabled the recent chat buttons upon 
				generating the assistant response
		Disabled the input box upon generating the 
				assistant response
		Improved the promt by removing the "chunks"
				on the assistant response
		Improved the promt by replacing the "CONTEXT"
				by "documents" on the assistant
				response
		Trasferred the time stamp on the right side
				of the bubble (user and assistant)
		Improved the apperance of input box
		Improved the chat conversation display
		Improved the apperance of the apperance of
				new, clear, and recent chat buttons
		Fixed the horizontal size of the side bar
		Interchanged the position of the user and
				assistant in the conversation area
		Improved the appearance of conversation
				display, user and assistant occupy
				only 75% of the display screen.
		Changed the Welcome message
		Changed the message in the input box
version 0.5	impprovement
		Improved clean_text() regex logic (preserves 
				paragraph breaks, case-insensitive 
				page-number matching); replaced print() 
				with proper logger; added empty-document 
				filtering in clean_documents().
		Improved the logger.py (Fixed UTF-8 
				encoding crash in logger; added log rotation 
				(5MB, 3 backups); improved log format and 
				cleaned up code structure.) Migrated deprecated 
				st.components.v1.html to st.iframe in copy 
				button component.
		Resolved the command line errors and warnings
				related to logger
		Changed the Page title	
		Added search functionality in recent chat
		Added stop button functionality
		Added action buttons (copy, like, dislike, 
				and regenerate)
		Removed the dead code, optimized the redundant code.
		Improved the prompt and added
version 0.6	impprovement 
		Added the view functionality on the acquired sources
		Transferred the styling of the UI in separate file
		Improved the quality and consistency of result
		Added the document deletion detection
		Added the empty string guard on input box
version 0.7 improvement
		Added guard for long query that consumed the token
		Improved the stopped button functionality during
				mid-stream
		Improved the source display filename truncation if
				the filename is too long
		Added the feature that the recent chat persist
				upon page reloading
		Added the error message for backend and reference 
				source failure.
				 
		
