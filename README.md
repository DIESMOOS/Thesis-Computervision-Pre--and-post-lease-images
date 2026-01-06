# Information Studies (IS) Master Thesis LateX Template
This is the official LateX template for your Information Studies and Data Science MSc thesis. It includes all the files necessary for writing a great thesis and should streamline the process of formatting such that you can focus on what is important, the content.

The folder structure follows the expected thesis structure:
- **Abstract:** A summary of results should be included. Avoid citations. Maximum length is 200 words.
- **Introduction:** Mention scientific field, problem statement, research gap and each research question. 
- **Related Work:** Your work needs to be grounded and compared to earlier work and the state-of-the-art.
Write about your related work here.
- **Methodology:** Focus on what you add to the existing method. Explain what you will do and why (and how).
Write about your methodology here. Here you should also present all the settings used in your experiments and your experimental setup. Everything you do should be reproducible.
- **Results:** Give the outcomes for each research question in the form of a table or graphic (with caption).
Write about your results here.
- **Discussion:** Compare your results with the state-of-the-art and reflect upon the results and limitations of the study.
Write your discussion here. Furthermore, give an outlook on what could be added to your work or what further research it could enable.
- **Conclusion:** Answer each research question and go into how the limitations of the study qualify the conclusion.

Obviously, this structure is flexible, but in general this is how not only a thesis but also papers and in general great research work is strucutred.

## Importing this template to Overleaf
Simply download the `latex-template.zip` which you can find in the root folder of this repository and head over to Overleaf. There you can create a new `Upload Project` which will ask you to upload a zip file. Select the `latex-template.zip` you have downloaded and that's it! We might be adding the template to the Overleaf Gallery but for now, this is the preferred workflow.

## Working offline
You can use Overleaf to work offline as well. However, this will require you to install a LaTeX compiler yourself on your machine and use Overleaf's `git` or `GitHub` functionality. Furthermore, you will need an editor such as Visual Studio Code for example. If this mode of operation interests you, feel free to head over to: https://www.overleaf.com/learn/how-to/Working_Offline_in_Overleaf

## In Overleaf
Once you have imported the template in Overleaf, you should switch your main `tex` file to the one you are currently working on. You can do so by going to the `Menu` (top left) and select under `Main document` one of the three files in `setup`. The content you want to generate should be added to the respective `tex` files in the `sections` folder. Files in `setup`, `document-classes`, and the `latexmkrc` should not be altered without permission from the respective supervisor, examiner, and thesis coordinator as the configured layout is the one you are supposed to be using (and submitting with).

