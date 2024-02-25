from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

if __name__ == "__main__":
    load_dotenv()

    information = """
    Gregory House is a fictional character and the titular protagonist of the American medical drama series House. Created by David Shore and portrayed by English actor Hugh Laurie, he leads a team of diagnosticians and is the Head of Diagnostic Medicine at the fictional Princeton-Plainsboro Teaching Hospital in Princeton, New Jersey.[1] House's character has been described as a misanthrope, cynic, narcissist, and curmudgeon, the last of which terms was named one of the top television words of 2005 in honor of the character.[2]

    In the series, the character's unorthodox diagnostic approaches, radical therapeutic motives, and stalwart rationality have resulted in much conflict between him and his colleagues.[3] House is also often portrayed as lacking sympathy for his patients, a practice that allows him time to solve pathological enigmas. The character is partly inspired by Sherlock Holmes.[4][5] A portion of the show's plot centers on House's habitual use of Vicodin to manage pain stemming from leg infarction involving his quadriceps muscle some years earlier, an injury that forces him to walk with a cane. This dependency is also one of the many parallels to Holmes, who was a habitual user of cocaine and other drugs.[6]

    The character received generally positive reviews and was included in several "best of" lists.[7][8] Tom Shales of The Washington Post called House "the most electrifying character to hit television in years".[9] For his portrayal, Laurie won various awards, including two Golden Globe Awards for Best Actor in a Television Series – Drama, two Screen Actors Guild Awards for Best Actor from Drama Series, two Satellite Awards for Best Actor in a Television Series – Drama, two TCA Awards for Individual Achievement in Drama, and has received a total of six Primetime Emmy Award nominations for Outstanding Lead Actor in a Drama Series.
  """

    summary_template = """
    Given the information {information} about the person I want:
    1. Two interesting facts about them.
  """

    # receives template variables and the template
    summary_promot_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_promot_template)

    response = chain.invoke(input={"information": information})

    print(response)
