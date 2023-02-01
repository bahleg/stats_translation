import json, fire, os, re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
ru_re = re.compile('[А-ЯЁа-яё]')

tag_re = re.compile('<.*?>')

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

def corpus_translate(text):
    sents = sent_tokenize(text)
    buf = []
    for s in sents:
        if len(list(ru_re.findall(s)))>0:#0.5*len(s):
            buf.append(my_translate(s))
        else:
            buf.append(s)
    return ' '.join(buf)#+'\n'
    
def translate_code(code, dest_language='en'):
    
    print ([code])
    sents = code.splitlines()
    buf = []
    for s in sents:
        if s.strip().startswith('#') and ru_re.search(s):
            buf.append('#' + my_translate(s.strip().lstrip('#')))
        else:
            buf.append(s)
    #print (['\n'.join(buf)])
   
    return '\n'.join(buf)+'\n'
    

def my_translate(x):
    tokens = tokenizer([x], return_tensors='pt')
    
    return tag_re.sub(' ', tokenizer.decode(*model.generate(**tokens)).strip())
    
    
def translate_markdown(text, dest_language='en'):
    
    # Regex expressions
    MD_CODE_REGEX='```[a-z]*\n[\s\S]*?\n```'
    CODE_REPLACEMENT_KW = 'xx_markdown_code_xx'

    MD_LINK_REGEX="\[[^)]+\)"
    LINK_REPLACEMENT_KW = 'xx_markdown_link_xx'

    # Markdown tags
    END_LINE='\n'
    IMG_PREFIX='!['
    HEADERS=['### ', '###', '## ', '##', '# ', '#'] # Should be from this order (bigger to smaller)

     # Inner function to replace tags from text from a source list
    def replace_from_list(tag, text, replacement_list):
        list_to_gen = lambda: [(x) for x in replacement_list]
        replacement_gen = list_to_gen()
        return re.sub(tag, lambda x: next(iter(replacement_gen)), text)

    # Create an instance of Tranlator
    #translator = Translator()

    # Inner function for translation
    def translate(text):
        # Get all markdown links
        md_links = re.findall(MD_LINK_REGEX, text)

        # Get all markdown code blocks
        md_codes = re.findall(MD_CODE_REGEX, text)

        # Replace markdown links in text to markdown_link
        text = re.sub(MD_LINK_REGEX, LINK_REPLACEMENT_KW, text)

        # Replace links in markdown to tag markdown_link
        text = re.sub(MD_CODE_REGEX, CODE_REPLACEMENT_KW, text)

        # Translate text
        #text = translator.translate(text, src='ru', dest=dest_language).text
        text = corpus_translate(text)
    
        # Replace tags to original link tags
        text = replace_from_list('[Xx]'+LINK_REPLACEMENT_KW[1:], text, md_links)

        # Replace code tags
        text = replace_from_list('[Xx]'+CODE_REPLACEMENT_KW[1:], text, md_codes)

        return text

    # Check if there are special Markdown tags
    if len(text)>=2:
        if text[-1:]==END_LINE:
            return translate(text)+'\n'

        if text[:2]==IMG_PREFIX:
            return text

        for header in HEADERS:
            len_header=len(header)
            if text[:len_header]==header:
                return header + translate(text[len_header:])

    return translate(text)

#export
def jupyter_translate(fname, language='en', rename_source_file=False, print_translation=False):
    """
    TODO:
    add dest_path: Destination folder in order to save the translated files.
    """
    data_translated = json.load(open(fname, 'r'))

    skip_row=False
    for i, cell in tqdm(enumerate(data_translated['cells'])):
        for j, source in enumerate(cell['source']):
            if cell['cell_type']=='markdown':
                    if source[:4] != '<img':  # Don't translate cause
                    # of: 1. ``` -> ëëë 2. '\n' disappeared 3. image's links damaged
                        data_translated['cells'][i]['source'][j] = \
                            translate_markdown(source, dest_language=language)
            if cell['cell_type']=='code':
                data_translated['cells'][i]['source'][j] = \
                            translate_code(source, dest_language=language)
            
            if print_translation:
                print(data_translated['cells'][i]['source'][j])

    if rename_source_file:
        fname_bk = f"{'.'.join(fname.split('.')[:-1])}_bk.ipynb" # index.ipynb -> index_bk.ipynb

        os.rename(fname, fname_bk)
        print(f'{fname} has been renamed as {fname_bk}')

        open(fname,'w').write(json.dumps(data_translated))
        print(f'The {language} translation has been saved as {fname}')
    else:
        dest_fname = f"{'.'.join(fname.split('.')[:-1])}_{language}.ipynb" # any.name.ipynb -> any.name_pt.ipynb
        open(dest_fname,'w').write(json.dumps(data_translated))
        print(f'The {language} translation has been saved as {dest_fname}')

def markdown_translator(input_fpath, output_fpath, input_name_suffix=''):
    with open(input_fpath,'r') as f:
        content = f.readlines()
    content = ''.join(content)
    content_translated = translate_markdown(content)
    if input_name_suffix!='':
        new_input_name=f"{'.'.join(input_fpath.split('.')[:-1])}{input_name_suffix}.md"
        os.rename(input_fpath, new_input_name)
    with open(output_fpath, 'w') as f:
        f.write(content_translated)


if __name__ == '__main__':
    fire.Fire(jupyter_translate)
