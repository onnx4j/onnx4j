package org.onnx4j.exceptions;

public class ForwarderException extends RuntimeException {

	private static final long serialVersionUID = -9110594428291367746L;

	protected int code;
	
	public ForwarderException(String message) {
		super(message);
	}
	
	public int getCode() {
		return code;
	}

}
