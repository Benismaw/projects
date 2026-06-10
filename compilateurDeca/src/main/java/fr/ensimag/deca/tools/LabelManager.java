package fr.ensimag.deca.tools;

import fr.ensimag.ima.pseudocode.Label;

public class LabelManager {
    private int counter = 0;

    public Label uniqueLabel(String prefix) {
        return new Label(prefix + "_" + (counter++));
    }
}
