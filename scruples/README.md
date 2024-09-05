The Scruples Anecdotes
======================
This directory contains the Scruples Anecdotes.

Scruples provides code and data for understanding norms and subjectivity in
natural language. For more information, see the code repository
(https://github.com/allenai/scruples), or the paper: "Scruples: A Corpus of
Community Ethical Judgments on 32,000 Real-Life Anecdotes"
(https://arxiv.org/abs/2008.09094).

To cite this data or other aspects of the paper, use:

    @article{Lourie2020Scruples,
        author = {Nicholas Lourie and Ronan Le Bras and Yejin Choi},
        title = {Scruples: A Corpus of Community Ethical Judgments on 32,000 Real-Life Anecdotes},
        journal = {arXiv e-prints},
        year = {2020},
        archivePrefix = {arXiv},
        eprint = {2008.09094},
    }


Data Structure
--------------
Each anecdote from the Scruples Anecdotes has the following attributes:

  - **id**
    A unique identifier for the anecdote.
  - **post_id**
    The reddit ID for the post from which the anecdote was extracted.
  - **action**
    An action summarizing the anecdote extracted from the post's title. The
    action itself has following attributes:
    - **description**
      The textual description of the action.
    - **pronormative_score**
      A (noisy) estimate of how ethically neutral or positive the action is,
      counting how many community members rated the author as not in the wrong
      in the corresponding anecdote.
    - **contranormative_score**
      A (noisy) estimate of how ethically negative the action is, counting how
      many community members rated the author as in the wrong in the
      corresponding anecdote.
  - **title**
    The title of the anecdote.
  - **text**
    The text of the anecdote.
  - **post_type**
    `HISTORICAL` if the anecdote is something the author did, and
    `HYPOTHETICAL` if it's something the author is considering doing.
  - **label_scores**
    Scores counting how many community members expressed each label.
  - **label**
    The majority label.
  - **binarized_label_scores**
    Scores for the binarized label for the anecdote. `RIGHT` sums `OTHER`
    and `NOBODY` while `WRONG` sums `AUTHOR` and `EVERYBODY`.
  - **binarized_label**
    The majority binarized label.


Files
-----
This directory should contain the following files:

  - **README**
    This README file.
  - **train.scruples-anecdotes.jsonl**
    The training split.
  - **dev.scruples-anecdotes.jsonl**
    The development split.
  - **test.scruples-anecdotes.jsonl**
    The testing split.


Contact
-------
For more information, see the code repository
(https://github.com/allenai/scruples), or the paper: "Scruples: A Corpus of
Community Ethical Judgments on 32,000 Real-Life Anecdotes"
(https://arxiv.org/abs/2008.09094). Questions and comments may be addressed to
Nicholas Lourie.
