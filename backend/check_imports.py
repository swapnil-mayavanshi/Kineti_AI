try:
    from langchain.chains import create_retrieval_chain
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    try:
        import langchain
        print(f"Langchain version: {langchain.__version__}")
    except:
        print("Langchain not found")
