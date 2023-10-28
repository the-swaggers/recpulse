# What to begin with?

### Learn some theory

If you don't know about CNNs, we highly recommend studying them. Also, here will be required:
* Matrix multiplication
* Gradient

Those are basic math concepts, they will be needed.

### Set up your environment

To run the project on you local machine you'll need docker. Build the image byt the following command:

```
docker build - recpulse
```

to run a container use

```
docker run -v .:/recpulse -it recpulse
```

Note: if you don't use you bash in the project folder, write global path to the directory instead of `.`.

### Use "make"

In this project we've added makefile, so once you're in the docker container run

```
make help
```

to get further information.

### Tests, tests and even more tests

Please add tests to all the code you've added. This might seem boring at the first sight, but it might save a tonn of time in the future.

