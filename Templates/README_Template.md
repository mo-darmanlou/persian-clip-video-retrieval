# True Structure of a good readme for DLF final project



1. what is the purpose of this application you've      developed?
    first of all, say what exactly your code does in **just one sentence, clear and simple**
    i.e: 
    ## Classifing the age and the education history of people using SVM with 97% precision

    not more, **you have just 3 seconds to tell people what exactly you do**

    <img src="https://memegenerator.net/img/instances/85747097.jpg" alt="stop" width="40%">


2. show others how to run your code in easiest way: \
    suppose that somebody wants to use your code, or do some test, create a walk through steps for the guy to run your code, show the steps for at least two OS, windows + linux, if you have problem, contact your team lead but dont leave this part alone!
    <img src="https://memegenerator.net/img/instances/85746997.jpg" alt="stop" width="40%">
    

    if you can't show a simple way to run your code, you've wasted your time, and your program will be extincted like Philosoraptor, some samples for this part can be like this:

    LINUX:
    ```sh
    git clone https://github.com/my_account/my_project.git
    cd git_started
    conda activate <virtual_name>
    pip install -r requirements.txt
    python main.py 
    ```

    WINDOWS:
    ```POWERSHELL
    git clone https://github.com/my_account/my_project.git
    cd git_started
    conda activate <virtual_name>
    pip install -r requirements.txt
    python main.py
    ```

    DO NOT HARDCODE VARIABLES! instead make a var json/yaml file and let people change varaibles easily

    ```json
    {
        "src" : "./src_dir",
        "dst" : "./dst_dir",
        "preprocess_threshold" : 20,
        "model_trained_file" : "./model/SVM.txt"
        "template_image" : "./templates/part.png"
    }
    ```

    congurlations!! you've already saved 2hours of somebodyelses time!
    
    sometimes you have pushed something that doesn't prepared yet, in this case put and **under construction, for run please get back here AUG 26** mark on top of the readme and mention the date which it can be preapred

3. show the system you'r working at
    i.e.: this code ran on a system with below config:
    > RAM: 6GB \
    > CPU: 4cores, i5 \
    > GPU: N/A \
    > OS: Linux, ubuntu 18.04 \
    > Language: Python 3.6.9 \
    > Database: SQLite 0.14

    <img src="https://memegenerator.net/img/instances/85747062.jpg" alt="stop" width="40%">


4. bring the workflow or the tree of the code in an simple boxes flow/tree, i.e:


    ![resources](https://docs.google.com/drawings/d/e/2PACX-1vSjkEOmBjTEeF0nwyFarKV9CAPfwu3T1SP4GaftJnfOyQVF_ANUScGgMdVPN8tS14YTPLeM7K9px-Lh/pub?w=680&h=292)


    ![tree](https://docs.google.com/drawings/d/e/2PACX-1vQZEm2hVNRgzw28by0KiIJnGu2JeDZnviXOudee53lvZtHnlG99rA7dZEDWp0Cq311G7PSfMPYjUOjv/pub?w=306&h=250)

    <img src="https://memegenerator.net/img/instances/85747154.jpg" alt="Realize" width="40%">




5. bring some samples of the input/output here, sometimes I don't want to run the code, I just want to know what the input/output looks like, a screenshot, an image,...
    
    GUI SKETCH i.e: 
    

 

    ![input_sketch_gui](https://docs.google.com/drawings/d/e/2PACX-1vTXAZ1XdbHM2VZo7kA0vwA6owT6-9PwNCBRrbq-ZaULJM3kgCZiiPMqLmmKANwQzEl9BPr1MdMrSEGz/pub?w=300&h=447 )
    
    CURRENT:

    ![output_sketch_gui](https://docs.google.com/drawings/d/e/2PACX-1vR3RF_pBrOWAIjgcuscZs8AaxSvlpC7VmPOQ5neOAFGjwDsI34pocveOkinNVcj90bHFqKvAeVeaZ5B/pub?w=957&h=250)


6. if your application have a precision/recall, show it here, note that, if you have **false detection, bring a sample here!!**
the precision is the division of true/(false+true) **DO NOT TRUST CODE OUTPUT**

7. branch a new branch for yourself in a project and anytime commit your codes and changes and pushed them to git requlary

8. if you want to show a walk through, do it here, step by step, please create a `requirements.txt` file, sometimes you can use `pip freeze` \
    the file should contain the package and the version numpy==1.0.5

9. explain about the algorithms you've used, prefrerly in flowchart

    ![algorithm](https://docs.google.com/drawings/d/e/2PACX-1vTu0RBFnsQU9HsyBeEpK35LGzd5yCoLB-MlXkfHGX-1D_2Gla_37LBIgjn5ccrEPTcMfI8gh1l_x_VV/pub?w=1137&h=116) 

10. timing \
    in race of a team of horses and donkeis, donkey ones  make the team to lose, any code has some donkeys (bottlenecks) and some horses, what's matter is that with how much overhead we can reach the goal, so you should identified your donkeys of your code, use time moudles in your code and put it in as much as possible in your code those timing, finally plot the time diffrences, it will declare where the code lags

    ```python
    import time
    from matplotlib import pyplot as plt
    time_array = []
    time_array.append(time.time())
    # first block (like reading input)
    time_array.append(time.time())
    # second block (like conversion)
    time_array.append(time.time())
    #.
    #.
    #.
    # fifth block (like output writing)
    time_array.append(time.time())
    plt.plot(time_array)
    plt.show()
    ```

![bottleneck](https://docs.google.com/drawings/d/e/2PACX-1vT3WGMHqPxUGH4BZZJmaN2wNM9BT9UELWa61iSng3qc58vwxEIYkseT9YtlQ3X5QYNs7AH0vWgazC83/pub?w=682&h=273)


11. dataset mangment \
    as there might be a continous creation on datasets, 
    Data sets are usually created in multiples. These should be managed on Google Drive under a unique name. or the labeling system
    directory structure:
    ```
    datasets
    |   description.txt
    |   dataset_managment_sheet.xls (a google spread sheet)
    |
    |___{workpiece type}
        |
        |___{region}
            |
            |___unprocessed
            |   |
            |   |___{dataset name}
            |   |___{dataset name}
            |
            |___processed
                |
                |___{architecture}
                    |
                    |___{dataset name}
                    |___{dataset name}
                        |
                        |___training
                        |   |
                        |   |   readme (a google doc)
                        |   |___images
                        |   |___labels
                        |
                        |___validation
                            |
                            |___images
                            |___labels

    ```

    for better managment, there would be a spreadsheet which shows the elemnets, links, number of data, owner ...:

    | id | workspace_type | region | dataset_name | n_train_data | n_validation_data | is_proceessed | parent_id | created_at | created_by | link | descp |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 1 | bottle_milk_1 | side | original1 | 562 | 76 | False | NULL | 20211126 | @somebody | https://... | slack: backlog: |
    
    before any training the dataset should split manually to avoid mixture of validation data to training data in shuffling, in each dataset there would be a readme document which explain how the dataset is generated

12. conceptual inclusion realtions for each project \

    ![concpetual inclusion](https://docs.google.com/drawings/d/e/2PACX-1vRRCbyRQIMCc9SQPixxiPRfLH2wh4Faks92tR6fK7e4RvLgE95ATpccy34tHhxjWr9XyMyBEHfJ0GEO/pub?w=190&h=100) 


13. create an comprehend description here, anything left




---
your first step to work in an IT-based company is to mingle with git, so search the internet about git main commands and get familiar with git , always use git, always document what you've done and always be organized to set meaningful commit message

**no need to memorize all git commands!** 

just keep with yourself git cheet sheet (toungetwisting sentence :D ) like this link: [git cheet sheet](https://github.github.com/training-kit/downloads/github-git-cheat-sheet.pdf)
or this one: [git basic commands](https://www.hostinger.co.uk/tutorials/basic-git-commands)
maybe this: [gitlab docs](https://www.tutorialspoint.com/gitlab/gitlab_git_commands.htm)
or if you had no time, watch this: [git crash](https://www.youtube.com/playlist?list=PLriKzYyLb28nCh3jJLROcYBvj7ZO0l-3G)

