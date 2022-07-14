.. _contributing:

Contributing Guide
===================

Here's the simple guide on how to develop or contribute to
decanter-ai-core-sdk.



Reporting issues
-------------------
A great way to contribute to the project is to send a detailed report when you
encounter an issue. We always appreciate a well-written issue report, and will
thank you for it!

Check that our `GitHub issues`_. doesn't already include that problem or
suggestion before submitting an issue. If you find a match, you can use the
"subscribe" button to get notified on updates. Do not leave random "+1" or "I
have this too" comments, as they only clutter the discussion, and don't help
resolving it. However, if you have ways to reproduce the issue or have
additional information that may help resolving the issue, please leave
a comment.

.. _GitHub issues: https://github.com/MoBagel/decanter-ai-core-sdk/issues

Bug Reports
~~~~~~~~~~~~~
Bug reports are hugely important! Before you raise one, please make sure
have followed the steps bellow:

*   Confirm that the bug hasn’t been reported before. Duplicate bug reports
    are a huge drain on the time of other contributors, and should be avoided
    as much as possible.
*   Please follow the issue template of Bug report, and give detailed and
    clear desciption.

Feature Requests
~~~~~~~~~~~~~~~~~
If you believe there is a feature missing, feel free to raise a feature
request. Make sure there's no duplicate feature raised before, and follow
the issue template of Feature requests.



Code Contributions
--------------------

When contributing code, you’ll want to follow this checklist:

1.  Fork the repository and make changes on your fork in a feature branch:

    *   If it's a bug fix branch, name it XXXX-something where XXXX is the
        number of the issue.
    *   If it's a feature branch, create an enhancement issue to announce
        your intentions, and name it XXXX-something where XXXX is the number
        of the issue.

2.  Run the tests ``make test`` to confirm they all pass on your system. If
    they don’t, you’ll need to investigate why they fail. If you’re unable
    to diagnose this yourself, raise it as a bug report by following the
    guidelines in this document: Bug Reports.

3.  Write tests that demonstrate your bug or feature. Ensure that they fail.

4.  Write clean code to make your change. Always run ``pylint file.py`` on
    each changed file before committing your changes. ``make lint`` for all
    files.

5.  Run the entire test suite again, confirming that all tests pass including
    the ones you just added.

6.  Send a GitHub Pull Request to the main repository’s master branch. Pull
    request descriptions should be as clear as possible and include a
    reference to all the issues that they address.

Conventions
~~~~~~~~~~~~
*   Before anybody starts working on bug fix or feature, any significant
    improvement should be documented as a GitHub issue.

*   **Commit messages**: Short summary (max. 50 chars) written in the
    imperative, followed by an optional, more detailed explanatory text
    which is separated from the summary by an empty line.

*   **Pull requests**:

    -   Code review comments may be added to your pull request. Discuss,
        then make the suggested modifications and push additional commits
        to your feature branch.

    -   Must be cleanly rebased on top of master without multiple branches
        mixed into the PR.

    -   Before you make a pull request, squash your commits into logical units
        of work using ``git rebase -i`` and ``git push -f``

    -   Include an issue reference like Closes #XXXX or Fixes #XXXX in the
        pull request description that close an issue. Including references
        automatically closes the issue on a merge.


Documentation Contributions
-----------------------------
Documentation improvements are always welcome! The documentation files live in
the ``docs/`` of the codebase. They’re written in reStructuredText, and use
Sphinx to generate the full suite of documentation.

When contributing document, you’ll want to follow this checklist:

1.  Build html by running ``make html`` in ``docs/``

2.  Please do your best to follow the style of the documentation files. This
    means a soft-limit of 79 characters wide in your text files and a
    semi-formal, yet friendly and approachable, prose style. Run ``make lint``
    in ``docs/``

3.  When presenting Python code, use single-quoted strings
    ('hello' instead of "hello").

Package Release
--------------------

If you have the right permission, and you have twine installed, releasing a package is as simple as

1.  update the version in ``setup.py``

2.  ``make package``

3.  ``make release``
