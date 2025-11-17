import os
import streamlit as st
import base64
import threading
import time
import re  # Import regular expressions
from io import BytesIO
from pypdf import PdfMerger, PdfReader
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from collections import defaultdict
from langchain_google_genai import ChatGoogleGenerativeAI
# --- 1. Setup & Configuration ---

load_dotenv()
gemini_llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",api_key=os.getenv("GEMINI_API_KEY"))
class PagePairData(BaseModel):
    """Extracted data from a pair of pages."""
    roll_number_1: str = Field("N/A", description="Roll number from the first image (Image 1).")
    page_number_1: str = Field("N/A", description="Answer sheet page number from the first image (as a digit).")
    roll_number_2: str = Field("N/A", description="Roll number from the second image (Image 2).")
    page_number_2: str = Field("N/A", description="Answer sheet page number from the second image (as a digit).")

class SinglePageData(BaseModel):
    """Extracted data from a single page."""
    roll_number_1: str = Field("N/A", description="Roll number from the image.")
    page_number_1: str = Field("N/A", description="Answer sheet page number from the image (as a digit).")

# --- 2. Helper Functions ---

@st.cache_resource
def get_pair_vision_llm():
    try:
        llm=gemini_llm
        return llm.with_structured_output(PagePairData)
    except Exception as e:
        st.error(f"Error initializing PAIR LLM: {e}")
        st.stop()

@st.cache_resource
def get_single_vision_llm():
    try:
        llm = gemini_llm
        return llm.with_structured_output(SinglePageData)
    except Exception as e:
        st.error(f"Error initializing SINGLE LLM: {e}")
        st.stop()

def merge_pdfs_to_bytes(pdf_files):
    merger = PdfMerger()
    for pdf_file in pdf_files:
        pdf_file.seek(0)
        merger.append(pdf_file)
    pdf_bytes_io = BytesIO()
    merger.write(pdf_bytes_io)
    merger.close()
    pdf_bytes_io.seek(0)
    return pdf_bytes_io.getvalue()

def encode_pil_image_to_base64(image):
    with BytesIO() as buffered:
        image.save(buffered, format="JPEG", quality=75)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def create_2_page_llm_message(base64_image_1, base64_image_2):
    return [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """
                  Analyze these two images. Image 1 is the first, Image 2 is the second.
                    Your task is to extract the 'roll_number' and 'page_number' for both images following strict formatting rules.

                    ### RULES SECTION:
                    1. **DIGITS ONLY:** The extracted values must contain ONLY numeric digits (0-9). Do not include any alphabets, symbols, or special characters.
                       - Incorrect: "A52", "Roll: 12", "Page-1"
                       - Correct: "52", "12", "1"
                    
                    2. **NO ALPHANUMERIC:** If the text on the page says "B-10" or "No. 5", extract only the number "10" or "5". Ignore all letters.

                    3. **LEADING ZEROS:** You must remove any leading zeros from the numbers. Treat numbers mathematically.
                       - If the roll number is "01", return "1".
                       - If the page number is "005", return "5".
                       - If the roll number is "099", return "99".

                    4. **LOCATIONS:**
                       - 'roll_number' is typically located in the top-right area.
                       - 'page_number' is typically located at the bottom of the page.

                    5. **MISSING DATA:** If you absolutely cannot find a value after checking carefully, return 'N/A'.

                    ### TASKS:
                    For Image 1: Extract 'roll_number' and 'page_number' applying the rules above.
                    For Image 2: Extract 'roll_number' and 'page_number' applying the rules above.

                    Respond *only* with the JSON.
                    """
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_2}"}}
            ]
        )
    ]

def create_1_page_llm_message(base64_image_1):
    return [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """
                   Analyze this image.
                    Your task is to extract the 'roll_number' and 'page_number' following strict formatting rules.

                    ### RULES SECTION:
                    1. **DIGITS ONLY:** The extracted values must contain ONLY numeric digits (0-9). Do not include any alphabets, symbols, or special characters.
                       - Incorrect: "A52", "Roll: 12", "Page-1"
                       - Correct: "52", "12", "1"
                    
                    2. **NO ALPHANUMERIC:** If the text on the page says "B-10" or "No. 5", extract only the number "10" or "5". Ignore all letters.

                    3. **LEADING ZEROS:** You must remove any leading zeros from the numbers. Treat numbers mathematically.
                       - If the roll number is "01", return "1".
                       - If the page number is "005", return "5".
                       - If the roll number is "099", return "99".

                    4. **LOCATIONS:**
                       - 'roll_number' is typically located in the top-right area.
                       - 'page_number' is typically located at the bottom of the page.

                    5. **MISSING DATA:** If you absolutely cannot find a value after checking carefully, return 'N/A'.

                    ### TASKS:
                    Extract 'roll_number' and 'page_number' applying the rules above.

                    Respond *only* with the JSON.
                    """
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_1}"}}
            ]
        )
    ]

# --- 3. Threading, Retry Logic & LLM Calls ---

def invoke_with_retry(llm, message, max_retries=3):
    """Invokes LLM with exponential backoff on 5xx errors."""
    for i in range(max_retries):
        try:
            return llm.invoke(message)
        except Exception as e:
            if "503" in str(e) or "500" in str(e): # Check for 5xx errors
                st.warning(f"Got 5xx error, retrying in {2**i}s... (Attempt {i+1}/{max_retries})")
                time.sleep(2**i) # Exponential backoff: 1s, 2s, 4s
            else:
                raise e
    raise Exception(f"Failed to call LLM after {max_retries} attempts.")


def process_page_pair_thread(
    pdf_index_1, pil_image_1, 
    pdf_index_2, pil_image_2, 
    llm, semaphore, results_list
):
    base64_image_1 = None
    base64_image_2 = None
    try:
        semaphore.acquire()
        base64_image_1 = encode_pil_image_to_base64(pil_image_1)
        base64_image_2 = encode_pil_image_to_base64(pil_image_2)
        message = create_2_page_llm_message(base64_image_1, base64_image_2)
        response = invoke_with_retry(llm, message)
        results_list.append((pdf_index_1, response.roll_number_1, response.page_number_1))
        results_list.append((pdf_index_2, response.roll_number_2, response.page_number_2))
    except Exception as e:
        results_list.append((pdf_index_1, "Error", str(e)))
        results_list.append((pdf_index_2, "Error", str(e)))
    finally:
        del base64_image_1, base64_image_2
        semaphore.release()

def process_single_page_thread(
    pdf_index_1, pil_image_1, 
    llm, semaphore, results_list
):
    base64_image_1 = None
    try:
        semaphore.acquire()
        base64_image_1 = encode_pil_image_to_base64(pil_image_1)
        message = create_1_page_llm_message(base64_image_1)
        response = invoke_with_retry(llm, message)
        results_list.append((pdf_index_1, response.roll_number_1, response.page_number_1))
    except Exception as e:
        results_list.append((pdf_index_1, "Error", str(e)))
    finally:
        del base64_image_1
        semaphore.release()

# --- 4. Streamlit App Main Logic (Unchanged) ---

st.set_page_config(layout="wide")
st.title("🚀 Robust PDF Student Sorter (with Retry)")
st.markdown("Handles odd pages and API errors. Processes 2 pages per call.")

MAX_CONCURRENT_CALLS = 10
llm_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_CALLS)
IMAGE_BATCH_SIZE = 20 

llm_pair = get_pair_vision_llm()
llm_single = get_single_vision_llm()

uploaded_files = st.file_uploader(
    "Upload your PDF files (will be merged)", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button(f"Start Processing (Using {MAX_CONCURRENT_CALLS} parallel requests)"):
        
        overall_start_time = time.time()
        results_list = []
        
        with st.spinner("Processing..."):
            try:
                # ... (Steps 1, 2, and 3 are identical to the previous code) ...
                with st.status("1/3 - Merging PDFs..."):
                    merged_pdf_bytes = merge_pdfs_to_bytes(uploaded_files)
                with st.status("2/3 - Getting page count..."):
                    pdf_reader = PdfReader(BytesIO(merged_pdf_bytes))
                    total_pages = len(pdf_reader.pages)
                    del pdf_reader
                st.success(f"Merged PDFs: {total_pages} total pages found.")

                with st.status(f"3/3 - Processing {total_pages} pages...", expanded=True) as main_status:
                    threads = []
                    for batch_start_page in range(1, total_pages + 1, IMAGE_BATCH_SIZE):
                        batch_end_page = min(batch_start_page + IMAGE_BATCH_SIZE - 1, total_pages)
                        main_status.update(label=f"Converting pages {batch_start_page}-{batch_end_page} to images...")
                        
                        batch_images = convert_from_bytes(
                            merged_pdf_bytes, 
                            dpi=150, 
                            first_page=batch_start_page, 
                            last_page=batch_end_page
                        )
                        
                        main_status.update(label=f"Spawning threads for pages {batch_start_page}-{batch_end_page}...")

                        for i in range(0, len(batch_images), 2):
                            pdf_index_1 = batch_start_page + i
                            pil_image_1 = batch_images[i]
                            
                            if i + 1 < len(batch_images):
                                pdf_index_2 = batch_start_page + i + 1
                                pil_image_2 = batch_images[i+1]
                                thread = threading.Thread(
                                    target=process_page_pair_thread,
                                    args=(
                                        pdf_index_1, pil_image_1, 
                                        pdf_index_2, pil_image_2, 
                                        llm_pair, llm_semaphore, results_list
                                    )
                                )
                            else:
                                thread = threading.Thread(
                                    target=process_single_page_thread,
                                    args=(
                                        pdf_index_1, pil_image_1, 
                                        llm_single, llm_semaphore, results_list
                                    )
                                )
                            threads.append(thread)
                            thread.start()

                        for thread in threads:
                            thread.join()
                        
                        threads = []
                        del batch_images
                        st.write(f"Batch {batch_start_page}-{batch_end_page} complete.")
                    st.success("All batches processed!")
                
                overall_end_time = time.time()
                total_duration = overall_end_time - overall_start_time

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()

        # --- 5. Post-Processing & Display (MODIFIED) ---
        
        st.header(f"Processing Complete")
        st.metric("Total Time Taken", f"{total_duration:.2f} seconds")
        st.metric("Total Pages Processed", total_pages)
        st.metric("Total API Calls Made", (total_pages + 1) // 2)

        # Filter out "N/A" or "Error" results
        valid_results = []
        for (pdf_idx, roll, page) in results_list:
            if roll and roll not in ["N/A", "Error"] and page and page not in ["N/A", "Error"]:
                valid_results.append((pdf_idx, roll, page))

        if not valid_results:
            st.error("No valid data could be extracted. Check your PDF and prompt.")
            st.stop()

        # --- NEW LOGIC: Build ONE data structure first ---
        # student_data will be: {'52': [('3', 58), ('4', 59), ('1', 60), ('2', 61)], ...}
        student_data = defaultdict(list)
        for (pdf_idx, roll_num, sheet_page_num) in valid_results:
            student_data[roll_num].append((sheet_page_num, pdf_idx))
        
        # --- NEW LOGIC: Define the sorting function ---
        def safe_int_sort(page_tuple):
            """
            Extracts the first number found in the page string (e.g., 'Page 1', '1', 'page-2')
            and uses it for sorting.
            """
            # page_tuple is ('1', 60)
            page_str = str(page_tuple[0]) 
            match = re.search(r'\d+', page_str) # Find the first number
            if match:
                try:
                    # print(match.group(0))
                    return int(match.group(0))
                except ValueError:
                    # print('9999')
                    return 99999 # Fallback (put at end)
            else:
                # print('99999')
                return 99999 # Fallback (put at end)

        # --- 1. The Mapper (Roll -> PDF Indices in student's order) ---
        st.subheader("Student Page Mapper (Pages in Order)")
        st.markdown("This table shows all PDF pages for each student, sorted by the student's **answer sheet page number**.")
        
        display_map = []
        # Sort by roll number for the table
        for roll_num in sorted(student_data.keys()):
            # Get the list of pages, e.g., [('3', 58), ('4', 59), ('1', 60), ('2', 61)]
            pages_list = student_data[roll_num]
            # print('hello')
            # Sort it by the *student's page number* (the first item in the tuple)
            # sorted_list will be: [('1', 60), ('2', 61), ('3', 58), ('4', 59)]
            sorted_list = sorted(pages_list, key=safe_int_sort) 
            # print(sorted_list)
            # Extract just the PDF indices from this sorted list
            # ordered_pdf_indices will be: [60, 61, 58, 59]
            ordered_pdf_indices = [pdf_idx for (sheet_page, pdf_idx) in sorted_list]
            
            # Join them for display
            pages_str = ", ".join(map(str, ordered_pdf_indices))
            
            display_map.append({"Roll Number": roll_num, "PDF Pages (in student's order)": pages_str})
        
        st.dataframe(display_map, width="stretch")

        # --- 2. The Sorter (Per-Student View) ---
        st.subheader("Student Answer Sheet Sorter")
        st.markdown("Select a student to see their answer sheets in the correct order.")
        
        student_list = sorted(student_data.keys())
        selected_student = st.selectbox("Select Student Roll Number:", student_list)

        if selected_student:
            # We already have the data, just need to re-sort it for display
            pages = student_data[selected_student]
            sorted_pages = sorted(pages, key=safe_int_sort)
            
            display_sorted = []
            for (sheet_page_num, pdf_idx) in sorted_pages:
                display_sorted.append({
                    "Student's Page Number": sheet_page_num,
                    "Located at Main PDF Page": pdf_idx
                })
            
            st.dataframe(display_sorted, width='stretch')
