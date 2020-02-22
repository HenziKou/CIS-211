#FiveTwelve

FiveTwelve is a sliding tile game based on 2048, with a few changes.  2048 was itself based on an earlier game called 1024, which (so legend says) was inspired by an earlier sliding tile game.  

##Game Play

The game is played on a 4x4 grid. Initially tiles containing the value 2 are placed randomly on the grid.  In each turn, the player may move all the tiles left, right, up, or down.  Tiles will slide as far as possible in the indicated direction, stopping when they reach an edge of the grid or when they meet a tile with a different value.  If a tile meets another tile with the same value, it absorbs the other tile (adding the value of the other tile to its own, thereby doubling), and *continues moving*. 

The order of movement matters when more than one merge is possible.  Consider a move *right* in the following row: 
```
2 4 4 4
```
Should this produce
```
_ 2 4 8
```
or  
```
_ 2 8 4
```
? The rule is that the rightmost tile moves first for a move right, the leftmost tile moves first for a move left, etc., so this move produces
```
_ 2 4 8
```

After each move, a new tile with value 2 is placed in a random open square on the grid.  When there is no open square to place the new tile, the game is over and the player's score is the sum of all tiles on the board.  

##Differences from 2048

The game play should be familiar to those who have played 1024, but there are a few differences. 

* Cascading merges:  In 2048, a tile may merge with another tile at most once in one turn.  In FiveTwelve, a tile may absorb another and continue to move in the same direction.  Thus, if we start with 
```2 2 4 8```
the leftmost tile will merge with second tile to form 
```_ 4 4 8```
and then continue, producing 
```_ _ 8 8```
and then 
```_ _ _ 16```
all in one move. 
* Ineffective moves are permitted.  Consider the board

```
   2 4 8 16
   2 4 8 16
   _ _ 8 16
   _ _ 8 16
```
The player may choose to slide the tiles right, but no tile can move farther to the right.  In 2048, this move is not allowed (the player must choose a different direction).  In FiveTwelve, the move right is allowed, and has no effect except to cause a new tile with value 2 to be introduced in one of the empty spaces. 

* New tiles all have value 2. In 2048, there is a 10% probability of a new tile having value 4. 
* You can never win.  Because FiveTwelve is like life.  You just keep adding to your score until gameplay ends. 

##Known bugs and limitations
* You must click the FiveTwelve window with a pointing device to send keystrokes to the game.  

##Implementation notes: MVC

FiveTwelve follows a Model-View-Controller (MVC) organization or *design pattern*.   The model component (model.py) contains all the game logic and data structures.  The model component has no direct dependencies on the view or controller components, but each element of the model component permits registration of *listeners* and announces significant events to its listeners.

The view component is responsible for maintaining the visual representation of the game.  Most objects in the model component have peer objects (with a related class name) in the view component. The view objects are registered as *listeners* on their model component peers. They receive notifications of changes to the model, and make corresponding changes to the visual representation. 

The view component depends on (and imports) the model component.  View objects may inspect their peer model objects (e.g., obtaining the current value of a tile, or its current position).  The model component does not depend on the view component, does not import it, and will operate the same with zero views, one view, or multiple views. 

The *controller* part of the Model-View-Controller organization is currently combined with the main program in game_manager.py, with keystroke acquisition in keypress.py.  The controller depends on both the the model and view componments, and makes the initial connection bertween 

The current view component uses Zelle's graphics module (graphics/graphics.py), which in turn uses the TkInter graphics module that is included in Python distributions.  Keypress.py also depends on graphics.py, and thereby on TkInter.  A version of FiveTwelve that uses PyQt, PyGame, or another graphics/GUI layer will require an implementation of modules providing the same API as view.py and keypress.py.  It should be possible to make minimal changes to game_controller.py (just importing the different view modules) and no changes at all to model.py. 

Most changes to game logic in model.py should also be possible without changing view.py. For example, adopting the 2048 rule regarding merging (only one merge per tile per move) should require no change to view.py.  Adopting the 2048 rule regarding ineffective moves would require small changes to model.py and controller.py but no change to view.py or keypress.py.  This independence is the point of MVC organization. 

##What students must program

 We have provided the logic for sliding a tile in a given direction, encoded in a 'movement vector' (dx, dy), until it either reaches the edge of the grid, reaches another tile and stops, or reaches another matching tile and merges with it. Each of the four possible moves (slide left, slide right, slide up, slide down) requires triggering the slide for each tile, in the proper order. You will need to slide each tile in the proper order. Recall again that a slide right starting with
````
2 4 4 4
````
This should produce

````
_ 2 4 8
````
and not
````
_ 2 8 4
````

You can earn extra credit with one or more of the following changes to make FiveTwelve more like 2048. However, extra credit is available only if all other elements of gameplay are correctly implemented. If you make one of these changes but break something else, you will lose points for what you broke and not gain credit for what you fixed. In practical terms this means that you need to use the provided test suites and add to them as needed to guard against accidentally making changes you did not intend.

* Up to 5 points: Alter FiveTwelve scoring to be the same as 2048 scoring.
* Up to 5 points: Disable ineffective moves. If the user's input does not cause any tile to move, no new tile should be generated.
* Up to 5 points: As in 2048, a new tile should be generated with value 4 rather than value 2 with probability 0.1. (Use the Python library module 'random' to make the decision.)
* Up to 5 ponts: Disable cascading merges: When a tile bumps into another tile of equal value and merges with it, it stops sliding.

None of these changes requires much code, but they require careful study of the existing code to find the best approach to implementing the change. If you choose poorly, your code will be more complicated and it may break other parts of the program. The credit for these is labeled “Up to ... ” because full extra credit requires more than implementing the needed functionality. It is just as important that you implement it in a way that is as clean and readable and maintainable as possible. That's the hard part!
