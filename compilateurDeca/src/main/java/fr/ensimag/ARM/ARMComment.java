package fr.ensimag.ARM;


public class ARMComment extends ARMInstruction {
        
    private final String comment;

    public ARMComment(String comment) {
        this.comment = comment;
    }

    @Override
    public String toString() {
        return "@ " + comment;
    }
}
