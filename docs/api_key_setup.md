## Setting Up Environment Variables

Create a `.env` file in the root directory of the project and add the following variables:

```
OPEN_API_KEY=your_open_face_api_key
```

Replace `your_open_face_api_key` with your actual Hugging Face API key.

## Troubleshooting

- Ensure all dependencies are installed correctly.
- Verify that the `.env` file is correctly set up with your API key.
- Check for any error messages in the terminal and address them accordingly.

For further assistance, refer to the project's documentation or contact the maintainers.

...
# Load the token
  load_dotenv()
  sec_key = os.getenv("TOKEN_KEY")
...