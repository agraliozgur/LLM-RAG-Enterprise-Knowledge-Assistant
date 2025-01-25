import google.generativeai as genai
import time
import logging

# Configuration
API_KEY = "AIzaSyA1jOI59ix7vSFMY1fCP7ykqeGwLNVpZYM"


# Configuration
MODEL_NAME = "gemini-1.5-pro-latest"
MODEL_NAME = "gemini-1.5-flash-latest"
OUTPUT_FILENAME = "../data/raw/txt_files/synthetic_ekm_data.txt"
OUTPUT_FILENAME = "../data/raw/txt_files/synthetic_ekm_data_flash.txt"

# Setup logging
logging.basicConfig(
    # filename='synthetic_ekm_generator.log',
    filename='synthetic_ekm_generator2.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure the GenAI model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def generate_ekm_synthetic_data(system_prompt, prompt):
    """
    Generates a synthetic EKM data paragraph based on the provided prompt.

    Args:
        system_prompt (str): The system prompt guiding the model.
        prompt (str): Specific scenario or case study to generate.

    Returns:
        str or None: Generated synthetic EKM data paragraph or None if failed.
    """
    full_prompt = f"{system_prompt}\n\n{prompt}"
    
    try:
        response = model.generate_content(full_prompt)
        if response.text:
            paragraph = response.text.strip()
            logging.info("Successfully generated a paragraph.")
            return paragraph
        else:
            logging.warning("No text received from the model.")
            return None
    except Exception as e:
        logging.error(f"An error occurred during generation: {e}")
        return None

def save_synthetic_data_to_txt(data, filename=OUTPUT_FILENAME):
    """
    Saves the generated synthetic data to a .txt file, each paragraph on a new line.

    Args:
        data (str): Generated paragraph.
        filename (str): Name of the file to save data.
    """
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(data + "\n")
        logging.info(f"Data successfully saved to {filename}!")
    except Exception as e:
        logging.error(f"Failed to save data to {filename}: {e}")

def get_prompts():
    """
    Returns a list of diverse prompts targeting different sectors and departments.

    Returns:
        list: List of prompt strings.
    """
    return [
        "Create a synthetic case study about improving knowledge sharing processes for a global retail company in the Marketing department.",
        "Generate a synthetic case study on enhancing knowledge retention strategies for a mid-sized healthcare company in the Human Resources department.",
        "Develop a synthetic case study focusing on implementing EKM tools for a technology startup in the IT department.",
        "Create a synthetic case study about optimizing knowledge workflows for a financial services firm in the Operations department.",
        "Generate a synthetic case study on fostering innovation through knowledge management for a large manufacturing company in the Research and Development department.",
        "Create a synthetic case study about streamlining knowledge documentation for a multinational finance corporation in the Compliance department.",
        "Generate a synthetic case study on enhancing employee onboarding through knowledge management for a mid-sized technology firm in the Human Resources department.",
        "Develop a synthetic case study focusing on integrating EKM systems for a global manufacturing company in the Supply Chain department.",
        "Create a synthetic case study about improving cross-departmental knowledge transfer for a large healthcare provider in the IT department.",
        "Generate a synthetic case study on leveraging knowledge management to support remote teams for a technology company in the Product Development department.",
        "Create a synthetic case study about enhancing knowledge sharing among frontline employees in a hospitality company’s Customer Service department.",
        "Generate a synthetic case study on implementing EKM practices to improve legal compliance in a financial firm’s Legal department.",
        "Develop a synthetic case study focusing on knowledge management solutions for procurement processes in a large automotive company’s Procurement department.",
        "Create a synthetic case study about using knowledge management to drive strategic decision-making in a telecommunications company’s Executive department.",
        "Generate a synthetic case study on integrating AI-powered knowledge retrieval systems in an education institution’s IT department.",
        # Add more prompts as needed to cover different sectors and departments
    ]

def main():
    system_prompt = """
    You are an Enterprise Knowledge Management (EKM) expert. Generate realistic yet completely synthetic data for various companies across different industries and departments. Each data entry should be a well-structured, comprehensive paragraph that includes the following elements:

    - **Company Information**: Fictional name, Industry (e.g., Healthcare, Finance, Retail, Technology, Manufacturing, Education, Energy, Transportation, Hospitality, Pharmaceuticals, Automotive, Telecommunications), Employee Count (ranging between 50 to 5000)
    - **Department**: (e.g., Human Resources, IT, Marketing, R&D, Operations, Compliance, Supply Chain, Customer Service, Legal, Procurement, Product Development, Finance, Executive)
    - **Tools and Technologies Used**: List of EKM tools and technologies employed (e.g., Confluence, SharePoint, Jira, Slack, Microsoft Teams, Asana, Trello, Knowledge Graphs, AI-Powered Search Engines, Document Management Systems)
    - **Key Challenges Faced**: Specific knowledge management challenges related to the industry and department (e.g., knowledge sharing in remote teams, knowledge retention during high employee turnover, maintaining up-to-date documentation in fast-paced environments, cross-departmental knowledge transfer, managing tacit knowledge, ensuring data privacy and compliance, integrating EKM systems with existing workflows, fostering a culture of continuous learning)
    - **Implemented Solutions**: Strategies and tools implemented to address the challenges (e.g., adopting new EKM platforms, conducting regular knowledge-sharing sessions, implementing mentorship programs, automating documentation processes, integrating AI for knowledge retrieval, establishing knowledge repositories, developing training programs)
    - **Measured Outcomes**: Tangible results and improvements observed after implementing the solutions (e.g., percentage increases in knowledge sharing efficiency, reductions in project turnaround times, improvements in employee onboarding processes, enhancements in customer satisfaction scores, increases in innovation rates)

    Ensure that each entry is unique, covers different sectors and departments, and avoids repetition. The data should reflect a variety of problems and solutions relevant to each context. Each entry should be presented as a separate, standalone paragraph.
    """

    prompts = get_prompts()
    prompt_count = len(prompts)
    current_prompt_index = 0

    while True:
        current_prompt = prompts[current_prompt_index]
        attempt = 0
        max_attempts = 3
        wait_times = [60, 300, 300]  # Wait times in seconds: 1 min, 5 min, 5 min

        while attempt < max_attempts:
            logging.info(f"Generating data for prompt {current_prompt_index + 1}/{prompt_count}: {current_prompt}")
            generated_paragraph = generate_ekm_synthetic_data(system_prompt, current_prompt)

            if generated_paragraph:
                save_synthetic_data_to_txt(generated_paragraph)
                break  # Successful generation, move to next prompt
            else:
                attempt += 1
                logging.warning(f"Attempt {attempt} failed for prompt: {current_prompt}")
                if attempt < max_attempts:
                    wait_time = wait_times[attempt - 1]
                    wait_minutes = wait_time / 60
                    logging.info(f"Waiting for {wait_minutes} minute(s) before retrying...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to generate data for prompt after {max_attempts} attempts. Skipping to next prompt.")
                    break

        # Move to the next prompt
        current_prompt_index = (current_prompt_index + 1) % prompt_count

        # Wait for 1 minute before sending the next query
        logging.info("Waiting for 1 minute before sending the next query...\n")
        time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
        logging.info("Script terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.critical(f"An unexpected error occurred: {e}")
