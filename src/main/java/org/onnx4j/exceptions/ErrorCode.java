package org.onnx4j.exceptions;

public class ErrorCode {
	
	private String errorCode;
	
	public ErrorCode(String prefix, int code) {
		this.errorCode = prefix + code;
	}

	@Override
	public String toString() {
		return errorCode;
	}

}
