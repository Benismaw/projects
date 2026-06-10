package fr.ensimag.ARM;

public class ARMLabel extends ARMInstruction{
    
    private String name;

    public ARMLabel(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name + ":";
    }

    public String getName() {
        return name;
    }
}
