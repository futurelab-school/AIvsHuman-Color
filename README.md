# AIvsHuman
This is for the color mixing game \
Written by: Austin Martin & Clara Tamura

## Play online
[Play the game in your browser](https://futurelab-school.github.io/AIvsHuman-Color/) (no install needed; the web version lives in `docs/`).

# Rules
### How to Play
- Adjust the CMYK sliders to match the target color.
- Hit the submit button when you want to make a guess.
- Keep trying until you get under 5% error.
- Try to beat the AI before it matches the color.
- You are given 8 random colors that the AI is also given for reference.


### Rules
1. Don't use the internet
2. You get one try per color. (Don't re-run the cell or you will lose your attemps)
3. Don't change the code!

```
Run the cell below to get started!

(\_/)                (\_/)
( •_•)              (•_• )
/>  /> ~ Have fun ! <\  <\

```


## To run it on google colab
```python
  # Get the game from the repo
  !pip install git+https://github.com/futurelab-school/AIvsHuman-Color.git

  from AIvsHuman import Color_Game
  # Launch the Game
  Color_Game().launch_game()
```

[Launch in Google Colab ](https://colab.research.google.com/drive/1RXAaAFMR7oo8DhOtcNUZ3YASUA_4BR46?usp=sharing#scrollTo=ssjpdZ-Fxksg)
