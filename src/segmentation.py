import string
import pathlib 
import jieba
from zhon import hanzi
def segmentation(dirname,output_file_name):

    '''
    Input:dirname->name of the data directory
    output: file named output_file_name in dirname directory, each line is an individual directory segmented
            , splitted with "," with 1st being the directory name and 2nd being the file name

    segment files in article directory, I did not have time to write it recursively
    so the files has to be put into project_directory/dirname/SOME_OTHER_Directory/file.txt
    import jieba for segmentation
    string and zhon for removing the punctuations
    I figure certain numbers might matter for topic extraction so I did not remove numbers


    '''
    articles=pathlib.Path(__file__).parent.parent / dirname #get into src directory
    processed = open(pathlib.Path(__file__).parent.parent / dirname / output_file_name, "w",encoding='utf8')
    counter=0
    for articles_dir in articles.iterdir():  
        try:     
            for  article in articles_dir.iterdir():
                counter+=1

                if counter%500==0:
                    print(f"processed {counter} files")

                with article.open(encoding="utf8") as text:
                    lines=text.read().splitlines()
                    #article_name=[x.strip(' ') for x in article.name]
                    article_name=article.name
                    new_name=''.join(article_name.split())
                    
                    processed.write(articles_dir.name+'/' + new_name)#第一个词是标题，之后要用，有的标题里面有空格，去掉这些空格不然格式不对
                    
                    processed.write(" ")

                    for line in lines:
                        seg_list = jieba.cut(line)
                        for word in seg_list:
                            if (word not in string.punctuation) and (word not in hanzi.punctuation):
                                processed.write(word)
                                processed.write(" ")  
                    processed.write("\n")                         
        except:
            pass         
    processed.close()



segmentation("article","segmented.txt")

